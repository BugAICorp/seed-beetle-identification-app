from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .tasks import run_evaluation_task, run_mc_dropout_evaluation_task, retrain_model_task, refresh_database_task
from port_inspector_app.models import Image, SpecimenUpload, User, KnownSpecies, Genus, ValidClasses
from .forms import UserRegisterForm, SpecimenUploadForm, ConfirmIdForm, ResetPasswordForm, ResetRequestForm
from django.core import signing
from django.core.cache import cache
from django.core.files.base import ContentFile
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMessage
from django.core.exceptions import ValidationError
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.http import HttpResponse, JsonResponse
from .tokens import account_activation_token, reset_account_token
from port_inspector.celery import app
import redis
import io
import os
from PIL import Image as PILImage
import time

# Setup global redis connection
redis_connection = None


def get_redis_conn():
    global redis_connection
    if redis_connection is None:
        redis_url = "redis://localhost:6379/0"
        redis_connection = redis.from_url(
            redis_url,
            decode_responses=False
        )
    return redis_connection


def verify_email(request, user_id):
    if request.method == "POST":
        current_site = get_current_site(request)
        user = User.objects.get(pk=user_id)
        email = user.email
        subject = "Verify Email"
        message = render_to_string(
            "verify-email-message.html",
            {
                "request": request,
                "user": user,
                "domain": current_site.domain,
                "uid": urlsafe_base64_encode(force_bytes(user.pk)),
                "token": account_activation_token.make_token(user),
            },
        )
        email = EmailMessage(subject, message, to=[email])
        email.content_subtype = "html"
        email.send()
        return redirect("verify-email-done")
    return render(request, "verify-email.html")


def verify_email_done(request):
    return render(request, "verify-email-done.html")


def verify_email_confirm(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and account_activation_token.check_token(user, token):
        user.is_email_verified = True
        user.is_active = True
        if user.email.lower().strip().endswith("@usda.gov"):
            user.is_usda = True
        user.save()
        messages.success(request, "Your email has been verified.")
        return redirect("/login/")
    else:
        messages.warning(request, "The link is invalid.")
    return render(request, "verify-email-confirm.html")


def signup_view(request):
    if request.method == "POST":
        print("signup POST request received\n")
        next_page = request.GET.get("next")
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            password = form.cleaned_data.get("password")
            user.set_password(password)
            user.name = form.cleaned_data.get("name")
            user.save()
            new_user = authenticate(email=user.email, password=password)
            if new_user:
                return redirect("verify-email", user_id=user.user_id)
            else:
                print("Authentication failed")
            if next_page:
                return redirect(next_page)
            else:
                return redirect("verify-email", user_id=user.user_id)
        else:
            print("ERROR: Email already in use or passwords do not match\n")
    else:
        form = UserRegisterForm()
    context = {"form": form}
    return render(request, "signup.html", context)


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            if not user.is_email_verified:
                messages.error(request, "Please verify your email before logging in.")
                return redirect("/login/")  # or re-render with error
            login(request, user)
            return redirect("/upload/")
    # if user is already logged in, redirect
    elif request.user.is_authenticated:
        return redirect("/upload/")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})


def forgot_password(request):
    if request.method == "POST":
        form = ResetRequestForm(request.POST)
        if form.is_valid():
            user_email = form.cleaned_data.get("email")
            user = User.objects.get(email=user_email)
            return redirect("reset-password-sent", user_id=user.user_id)
    else:
        form = ResetRequestForm()
    context = {"form": form}
    return render(request, "forgot-password.html", context)


def reset_password(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and reset_account_token.check_token(user, token):
        if request.method == "POST":
            form = ResetPasswordForm(request.POST)
            if form.is_valid():
                new_password = form.cleaned_data.get("password")
                user.set_password(new_password)
                user.save()
                return redirect('/login/')
        else:
            form = ResetPasswordForm()
        return render(request, 'reset-password.html', {"form": form, "validLink": True})
    else:
        return render(request, "reset-password.html", {"validLink": False})


def reset_password_sent(request, user_id):
    if request.method == "POST":
        current_site = get_current_site(request)
        user = User.objects.get(pk=user_id)
        email = user.email
        subject = "Reset Password"
        message = render_to_string(
            "reset-email-message.html",
            {
                "request": request,
                "user": user,
                "domain": current_site.domain,
                "uid": urlsafe_base64_encode(force_bytes(user.pk)),
                "token": reset_account_token.make_token(user),
            },
        )
        email = EmailMessage(subject, message, to=[email])
        email.content_subtype = "html"
        email.send()
        return render(request, "reset-password-sent.html", {"email_sent": True})
    return render(request, "reset-password-sent.html", {"email_sent": False})


# log the user out and send them back to the upload page
def logout_view(request):
    logout(request)
    return redirect("/upload/")


@login_required(login_url='/login/')
def upload_image(request):
    upload_id = request.GET.get("upload_id")
    failed_param = request.GET.get("failed_views", "")
    failed_views = failed_param.split(",") if failed_param else []

    image_urls = {
        "frontal": None,
        "dorsal": None,
        "caudal": None,
        "lateral": None,
    }

    # Handle POST (user submitting new upload)
    if request.method == "POST":
        specimen_form = SpecimenUploadForm(request.POST, request.FILES)

        if specimen_form.is_valid():
            specimen = specimen_form.save(user=request.user)
            hashed_ID = signing.dumps(specimen.id, salt=settings.SALT_KEY)
            return redirect("results", hashed_ID=hashed_ID)  # go to a UNIQUE URL for the results

    else:
        specimen_form = SpecimenUploadForm()

    # Handle GET (display existing images for re-upload)
    if upload_id:
        try:
            upload = SpecimenUpload.objects.get(id=upload_id)
        except SpecimenUpload.DoesNotExist:
            upload = None

        if upload:
            image_urls = {
                "frontal": upload.frontal_image.image.url if upload.frontal_image else None,
                "dorsal": upload.dorsal_image.image.url if upload.dorsal_image else None,
                "caudal": upload.frontal_image.image.url if upload.frontal_image else None,
                "lateral": upload.caudal_image.image.url if upload.caudal_image else None,
            }

    return render(
        request,
        'upload_photo.html',
        {
            'form': specimen_form,
            "image_urls": image_urls,
            "failed_views": failed_views
        }
    )


@login_required(login_url='/login/')
def view_history(request):
    # create empty set of type SpecimenUpload
    specimen = SpecimenUpload.objects.filter(user=request.user).order_by('-upload_date')
    uploads = []
    for upload in specimen:
        hashed_ID = signing.dumps(upload.id, salt=settings.SALT_KEY)

        uploads.append((upload, hashed_ID))

    return render(
        request,
        'history.html',
        {
            'uploads': uploads,
            'max_uploads': settings.USER_MAX_UPLOADS,
        }
    )


def delete_upload_from_redis(upload_id: int):
    """ Deletes all cached images for a given SpecimenUpload from Redis. """
    redis_conn = get_redis_conn()
    if not redis_conn:
        return

    for view in ["lateral", "dorsal", "frontal", "caudal"]:
        key = f"upload:{upload_id}:{view}"
        try:
            redis_conn.delete(key)
        except redis.RedisError as e:
            print(f"Failed to delete {key} from Redis: {e}")


def store_upload_in_redis(upload: SpecimenUpload):
    """ Stores all images for a given SpecimenUpload in Redis. """
    redis_conn = get_redis_conn()
    if not redis_conn:
        return

    for view in ["lateral", "dorsal", "frontal", "caudal"]:
        img_field = getattr(upload, f"{view}_image")
        if img_field and img_field.image:
            try:
                img = PILImage.open(img_field.image.path).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                img_bytes = buf.getvalue()

                # Key format: "upload:<id>:<view>"
                key = f"upload:{upload.id}:{view}"
                redis_conn.set(key, img_bytes)
            except (redis.RedisError, IOError) as e:
                print(f"[Redis/Image Error] Failed to store {key}: {e}")
            finally:
                if 'img' in locals():
                    img.close()


def save_redis_images_to_disk(upload):
    """
    Save any cropped images in Redis back to the filesystem
    for the given SpecimenUpload.
    """
    redis_conn = get_redis_conn()
    if not redis_conn:
        return

    for view in ["lateral", "dorsal", "frontal", "caudal"]:
        key = f"upload:{upload.id}:{view}"
        try:
            img_bytes = redis_conn.get(key)
            if img_bytes:
                image_obj = getattr(upload, f"{view}_image")
                if image_obj and hasattr(image_obj, "image"):
                    # Overwrite the existing file
                    image_obj.image.save(
                        os.path.basename(image_obj.image.name),
                        ContentFile(img_bytes),
                        save=True
                    )
        except (redis.RedisError, IOError) as e:
            print(f"[Redis/Image Error] Failed to save {key} to disk: {e}")


def results_view(request, hashed_ID):
    # Try to get the upload
    try:
        upload_id = signing.loads(hashed_ID, salt=settings.SALT_KEY)
        upload = SpecimenUpload.objects.get(id=upload_id)
    except (SpecimenUpload.DoesNotExist, signing.BadSignature):
        # Invalid id/Upload does not exist
        messages.error(request, "The requested specimen does not exist or the link is invalid.")
        return redirect("upload")

    # Get user roles
    try:
        is_usda = request.user.is_usda
        is_special_status = request.user.is_special_status
    except AttributeError:
        is_usda = False
        is_special_status = False

    # If SpecimenUpload has not been evaluated yet, evaluate and store in db
    # Handle evaluation
    # Check if upload already has results
    if upload.task_status == "PENDING":

        try:
            store_upload_in_redis(upload)
        except Exception as e:
            print(f"[Redis Warning] Failed to store upload in Redis: {e}")

        if is_usda or is_special_status:
            task = run_mc_dropout_evaluation_task.delay(upload.id)

        else:
            task = run_evaluation_task.delay(upload.id)

        # store the Celery task ID with the upload
        print("DEBUG — Task Object:", task)
        print("DEBUG — Task ID:", getattr(task, "id", None))
        upload.task_id = task.id
        upload.task_status = "STARTED"
        upload.save()

        return render(
            request,
            "loading.html",
            {
                "hashed_ID": hashed_ID,
                "upload_id": upload.id
            }
        )

    # Handle running job
    if upload.task_status == "STARTED" or upload.task_status == "PROCESSING":
        return render(
            request,
            "loading.html",
            {
                "hashed_ID": hashed_ID,
                "upload_id": upload.id
            }
        )

    if upload.task_status == "FAILED_CROP":
        if not getattr(upload, "redis_cleaned", False):
            try:
                delete_upload_from_redis(upload.id)
            except Exception as e:
                print(f"[Redis Warning] Failed to delete upload from Redis: {e}")
            upload.redis_cleaned = True
            upload.save(update_fields=["redis_cleaned"])

        failed = getattr(upload, "failed_views", [])
        failed_param = ",".join(failed)
        return redirect(f"/upload/?upload_id={upload.id}&failed_views={failed_param}")

    # If task failed
    if upload.task_status == "FAILED":
        if not getattr(upload, "redis_cleaned", False):
            try:
                delete_upload_from_redis(upload.id)
            except Exception as e:
                print(f"[Redis Warning] Failed to delete upload from Redis: {e}")
            upload.redis_cleaned = True
            upload.save(update_fields=["redis_cleaned"])
        messages.error(request, "Evaluation failed. Please try again later.")
        return redirect("upload")

    # Status == COMPLETE
    if not getattr(upload, "redis_cleaned", False):
        try:
            # Save cropped images to disk first
            save_redis_images_to_disk(upload)
            # Then delete from Redis
            delete_upload_from_redis(upload.id)
        except Exception as e:
            print(f"[Redis Warning] Failed to delete upload from Redis: {e}")
        upload.redis_cleaned = True
        upload.save(update_fields=["redis_cleaned"])

    # Extract results from upload
    species_results = upload.species
    genus_result = upload.genus

    # Fetch species URLs from the database
    species_names = [species[0] for species in species_results if species and species[0]]
    species_data = KnownSpecies.objects.filter(species_name__in=species_names).values_list("species_name", "resource_link")
    species_dict = dict(species_data)

    # Fetch genus URL from the database
    genus_name = genus_result[0]
    genus_data = Genus.objects.filter(genus_name=genus_name).values_list("genus_name", "resource_link")
    genus_dict = dict(genus_data)

    # Build species results with URLs
    formatted_species_results = [
        {
            "species_name": species[0],
            "confidence_level": species[1],
            "resource_link": species_dict.get(species[0], "#"),  # Default to "#" if not found
        }
        for species in species_results
        if species and species[0]
    ]

    # Species are already sorted from evaluation method
    # Include the genus at the top
    formatted_species_results.insert(
        0,
        {
            "species_name": genus_name,
            "confidence_level": genus_result[1],
            "resource_link": genus_dict.get(genus_name, "#"),
        },
    )

    # Determine the most likely species (excluding genus)
    likely_species = formatted_species_results[1]["species_name"] if len(formatted_species_results) > 1 else "Unknown"

    image_urls = ["", "", "", ""]
    if upload:
        image_urls[0] = upload.frontal_image.image if upload.frontal_image is not None else "default_image.jpg"
        image_urls[1] = upload.dorsal_image.image if upload.dorsal_image is not None else "default_image.jpg"
        image_urls[2] = upload.caudal_image.image if upload.caudal_image is not None else "default_image.jpg"
        image_urls[3] = upload.lateral_image.image if upload.lateral_image is not None else "default_image.jpg"

    confirm_choices = [(name, name) for name in species_names] + [("Other", "Other")]

    # Confirm species form
    if request.method == "POST":
        confirm_form = ConfirmIdForm(request.POST, choices=confirm_choices)
        if confirm_form.is_valid():
            upload.final_identification = confirm_form.cleaned_data['choice']
            upload.save()  # Save new data to the database
            # TODO add some form of confirmation here
            print("IDENTIFIED AS: ", upload.final_identification)
    else:
        confirm_form = ConfirmIdForm(choices=confirm_choices)

    confirmed_species = upload.final_identification

    # Determine result message type
    warning_message = None

    genus_certain = upload.genus_status
    species_certain = upload.species_status
    # Case 1: Genus Certain + Species Certain
    if genus_certain and species_certain:
        if genus_result[0].split()[0] != formatted_species_results[1]["species_name"].split()[0]:
            warning_message = "ERROR: The predicted genus and species do not match. " \
                "Please see identification resources to help identify the specimen."
    # Case 2–4: Any combination with 'Uncertain'
    elif not (genus_certain and species_certain):
        warning_message = "WARNING: Model is uncertain about the identification. " \
            "Please see identification resources to help identify the specimen."

    return render(
        request,
        "results.html",
        {
            "species_results": formatted_species_results[:6],  # Ensure only 5 species + 1 genus are displayed
            "upload_id": upload.id if upload else "DELETED",
            "likely_species": likely_species,
            "confirmed_species": confirmed_species,
            "image_urls": image_urls,
            "confirm_form": confirm_form,
            "is_usda": is_usda,
            "is_special_status": is_special_status,
            "species_stat": upload.species_status,
            "genus_stat": upload.genus_status,
            "species_uncert": upload.species_uncertainty,
            "genus_uncert": upload.genus_uncertainty,
            "species_conf_label": upload.species_confidence_label,
            "genus_conf_label": upload.genus_confidence_label,
            "warning_message": warning_message
        },
    )


def upload_status(request, upload_id):
    """
    Returns JSON indicating whether the Celery task is complete.
    """
    try:
        upload = SpecimenUpload.objects.get(id=upload_id)
        return JsonResponse({"task_status": upload.task_status})
    except SpecimenUpload.DoesNotExist:
        return JsonResponse({"task_status": "FAILED"})


def notify_unknown(request):
    if request.method == "POST":
        user = request.user
        if user.is_usda:
            send_to_email = "bruchinaiapp@gmail.com"
            results_page_url = request.META.get("HTTP_REFERER", "/history/")
            subject = "Port Inspector App - Unknown Species Uploaded"
            message = f"""
            <p>Hello,</p>
            <p>A user was unable to identify a specimen.</p>
            <p>Confirm the identification here: <a href="{results_page_url}">{results_page_url}</a></p>
            <p>Best,</p>
            <p>Port Inspector App Team</p>
            """
            email = EmailMessage(subject, message, to=[send_to_email])
            email.content_subtype = "html"
            email.send()

    return redirect("/history/")


@login_required(login_url='/login/')
def profile_view(request):
    user = request.user
    context = {
        "email": user.email,
        "name": user.name,
        "usda_user": user.is_usda
    }
    return render(request, 'profile.html', context)


@staff_member_required
def mass_upload_images(request):
    # Handles uploads of images from admin for the training database
    if request.method == "POST":
        images = request.FILES.getlist('images')
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        upload_dir = os.path.join(base_dir, "dataset")
        os.makedirs(upload_dir, exist_ok=True)

        for image in images:
            with open(os.path.join(upload_dir, image.name), 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

        # Enqueue the database refresh as a background task
        refresh_database_task.delay()

        return redirect("/admin/")


@staff_member_required
def retrain_models_thread(request):
    """
    Triggers model retraining as a Celery task from the admin interface.
    Prevents duplicate tasks by checking a cache flag.
    """
    status = cache.get("retrain_status")

    if status == "running":
        return redirect("/admin/")

    # Mark retraining as running
    cache.set("retrain_status", "running")

    # Enqueue the Celery task
    retrain_model_task.delay()

    return redirect("/admin/")


def check_retrain_status(request):
    """checks cache status"""
    status = cache.get("retrain_status", "idle")
    return JsonResponse({"status": status})


def home_view(request):
    return render(request, 'index.html')


def about_view(request):
    valid_classes = ValidClasses.objects.all()
    return render(request, 'about.html', {'valid_classes': valid_classes})
