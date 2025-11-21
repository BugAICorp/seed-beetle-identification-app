""" tasks.py """

from django.core.cache import cache
from PIL import Image as PILImage
import redis
import io
from celery import shared_task
from port_inspector_app.models import SpecimenUpload

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


def crop_views_from_redis(upload_id, cropper):
    """
    Attempt YOLO-based beetle cropping for each available view on a SpecimenUpload.

    Args:
        upload_id (int): SpecimenUpload id
        cropper (BeetleCropper): The YOLO cropping utility.

    Returns:
        failed_views (list[str]): views where no beetle was detected or cropping failed
    """
    redis_conn = get_redis_conn()
    if not redis_conn:
        return [], 0  # Redis not configured, skip

    failed_views = []

    for view in ["lateral", "dorsal", "frontal", "caudal"]:

        key = f"upload:{upload_id}:{view}"
        img_bytes = redis_conn.get(key)
        if not img_bytes:
            continue

        cropped_img = None
        try:
            img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            cropped_img = cropper.crop_beetle(img)
            if cropped_img is None:
                failed_views.append(view)
                continue

            # Replace upload with cropped image in memory
            img_bytes = io.BytesIO()
            cropped_img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            redis_conn.set(key, img_bytes.read())

        except Exception as e:
            print(f"[Crop Error] Failed to crop view {view} for upload {upload_id}: {e}")
            failed_views.append(view)

        finally:
            img.close()
            if cropped_img is not None:
                cropped_img.close()

    return failed_views


def get_view_paths(upload):
    """
    Extract filesystem paths for all four potential view images.

    Args:
        upload (SpecimenUpload): Upload instance with linked image fields.

    Returns:
        tuple(str | None): Paths to lateral, dorsal, frontal, and caudal images.
                           Any view not provided returns None.
    """
    return (
        upload.lateral_image.image.path if upload.lateral_image else None,
        upload.dorsal_image.image.path if upload.dorsal_image else None,
        upload.frontal_image.image.path if upload.frontal_image else None,
        upload.caudal_image.image.path if upload.caudal_image else None,
    )


@shared_task(bind=True)
def run_evaluation_task(self, upload_id):
    """
    Celery task to asynchronously run the default model evaluation.
    """
    from beetle_detection.beetle_cropper import BeetleCropper
    from beetle_detection.species_eval import evaluate_images

    cropper = BeetleCropper(threshold=0.8)

    failed_views = crop_views_from_redis(upload_id, cropper)

    if len(failed_views) > 0:
        SpecimenUpload.objects.filter(id=upload_id).update(
            task_status="FAILED_CROP",
            failed_views=failed_views
        )
        return

    SpecimenUpload.objects.filter(id=upload_id).update(task_status="PROCESSING")

    try:
        print("Evaluating upload:", upload_id)
        s, g = evaluate_images(upload_id)

        result = {
            "top_5_species": s,
            "species_uncertainty": 0.0,
            "species_status": True,
            "top_genus": g,
            "genus_uncertainty": 0.0,
            "genus_status": True,
        }

        if upload_id:
            SpecimenUpload.objects.filter(id=upload_id).update(
                species=s,
                genus=g,
                species_uncertainty=result["species_uncertainty"],
                genus_uncertainty=result["genus_uncertainty"],
                species_status=result["species_status"],
                genus_status=result["genus_status"],
                task_status="COMPLETE",
                failed_views=failed_views
            )

        return result

    except Exception:
        if upload_id:
            SpecimenUpload.objects.filter(id=upload_id).update(
                task_status="FAILED",
            )
        raise


@shared_task(bind=True)
def run_mc_dropout_evaluation_task(self, upload_id):
    """
    Celery task to asynchronously run the Monte Carlo Dropout model evaluation.
    """
    from beetle_detection.beetle_cropper import BeetleCropper
    from beetle_detection.species_eval import evaluate_mc_dropout

    cropper = BeetleCropper(threshold=0.8)

    failed_views = crop_views_from_redis(upload_id, cropper)

    if len(failed_views) > 0:
        SpecimenUpload.objects.filter(id=upload_id).update(
            task_status="FAILED_CROP",
            failed_views=failed_views
        )
        return

    SpecimenUpload.objects.filter(id=upload_id).update(task_status="PROCESSING")

    try:
        s, g, s_uncert, s_status, g_uncert, g_status = evaluate_mc_dropout(upload_id)

        result = {
            "top_5_species": s,
            "species_uncertainty": s_uncert,
            "species_status": s_status,
            "top_genus": g,
            "genus_uncertainty": g_uncert,
            "genus_status": g_status,
        }

        if upload_id:
            SpecimenUpload.objects.filter(id=upload_id).update(
                species=s,
                genus=g,
                species_uncertainty=s_uncert,
                genus_uncertainty=g_uncert,
                species_status=s_status,
                genus_status=g_status,
                task_status="COMPLETE",
                failed_views=failed_views
            )

        return result

    except Exception:
        if upload_id:
            SpecimenUpload.objects.filter(id=upload_id).update(
                task_status="FAILED",
            )
        raise


@shared_task(bind=True)
def retrain_model_task(self):
    """
    Asynchronous Celery task to retrain ML models.
    Updates cache with status to prevent duplicate tasks.
    """
    from beetle_detection.species_eval import retrain_models

    try:
        retrain_models()
        cache.set("retrain_status", "complete", timeout=30)
        return "Retraining finished successfully"
    except Exception:
        cache.set("retrain_status", "failed", timeout=30)
        return "Retraining failed"


@shared_task(bind=True)
def refresh_database_task(self):
    """
    Offloads the refresh_database() call to a background Celery worker.
    """
    from beetle_detection.species_eval import refresh_database
    try:
        refresh_database()
        return "Database refresh completed successfully"
    except (IOError, ConnectionError) as e:  # Only retry on transient errors
        raise self.retry(exc=e, countdown=60)
    except Exception:
        # Don't retry on other exceptions
        raise
