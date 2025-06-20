from django.contrib import admin
from django.urls import path
from django.utils.html import format_html
from django.utils.safestring import mark_safe
import os
import uuid
from io import BytesIO
from PIL import Image as img

# Register your models here.
from .models import SpecimenUpload, Image, KnownSpecies, Genus, TrainingDatabase, User, ValidClasses

admin.site.register(KnownSpecies)
admin.site.register(Genus)
admin.site.register(TrainingDatabase)
admin.site.register(User)


@admin.action(description="Transfer to Training Database")
def add_to_trainingdb(modeladmin, request, queryset):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    dir = os.path.join(base_dir, "dataset")
    for obj in queryset:
        # Check that the object is not already in the dataset and has been validated
        if obj.in_training:
            continue
        if not obj.is_validated:
            continue

        # Generate a new id for the specimen and parse its classication
        new_uid = str(uuid.uuid4())
        parsed_field = obj.final_identification.split(' ')
        if len(parsed_field) != 2:
            continue
        # Checks to see if the classification is meant to be trained with
        if not ValidClasses.objects.filter(genus=parsed_field[0]).exists() and not ValidClasses.objects.filter(species=parsed_field[1]).exists():
            continue 
        # Update object to remember it has been added already
        obj.in_training = True
        obj.save()

        # Check each of four possible images and add to training database
        if obj.frontal_image and obj.frontal_image.image:
            image_binary = obj.frontal_image.image.read()
            TrainingDatabase.objects.create(
                genus=parsed_field[0],
                species=parsed_field[1],
                uniqueid=new_uid+"FRON",
                view="FRON",
                specimenid=new_uid,
                image=image_binary
            )
            pil_img = img.open(BytesIO(image_binary)).convert("RGB")
            pil_img.save(os.path.join(dir, f"{parsed_field[0]} {parsed_field[1]} {new_uid} USER FRON.jpg"), format="JPEG")
        if obj.caudal_image and obj.caudal_image.image:
            image_binary = obj.caudal_image.image.read()
            TrainingDatabase.objects.create(
                genus=parsed_field[0],
                species=parsed_field[1],
                uniqueid=new_uid+"CAUD",
                view="CAUD",
                specimenid=new_uid,
                image=image_binary
            )
            pil_img = img.open(BytesIO(image_binary)).convert("RGB")
            pil_img.save(os.path.join(dir, f"{parsed_field[0]} {parsed_field[1]} {new_uid} USER CAUD.jpg"), format="JPEG")
        if obj.dorsal_image and obj.dorsal_image.image:
            image_binary = obj.dorsal_image.image.read()
            TrainingDatabase.objects.create(
                genus=parsed_field[0],
                species=parsed_field[1],
                uniqueid=new_uid+"DORS",
                view="DORS",
                specimenid=new_uid,
                image=image_binary
            )
            pil_img = img.open(BytesIO(image_binary)).convert("RGB")
            pil_img.save(os.path.join(dir, f"{parsed_field[0]} {parsed_field[1]} {new_uid} USER DORS.jpg"), format="JPEG")
        if obj.lateral_image and obj.lateral_image.image:
            image_binary = obj.lateral_image.image.read()
            TrainingDatabase.objects.create(
                genus=parsed_field[0],
                species=parsed_field[1],
                uniqueid=new_uid+"LATE",
                view="LATE",
                specimenid=new_uid,
                image=image_binary
            )
            pil_img = img.open(BytesIO(image_binary)).convert("RGB")
            pil_img.save(os.path.join(dir, f"{parsed_field[0]} {parsed_field[1]} {new_uid} USER LATE.jpg"), format="JPEG")


@admin.register(SpecimenUpload)
class SpecimenUploadAdmin(admin.ModelAdmin):
    """
    Add formatting for the specimen upload view on the admin page
    """
    list_display = ('id', 'formatted_genus', 'formatted_species', 'final_identification', 'display_all_images')
    list_filter = ('final_identification', 'is_validated', )
    readonly_fields = ['display_all_images', 'formatted_genus', 'formatted_species']
    fields = ('display_all_images', 'formatted_genus', 'formatted_species', 'final_identification', 'is_validated')
    actions = [add_to_trainingdb]

    def formatted_genus(self, obj):
        """
        Format the genus column to be more admin reader friendly
        """
        if isinstance(obj.genus, (list, tuple)) and len(obj.genus) == 2:
            name, confidence = obj.genus
            return f"{name}: {confidence:.2f}%"
        return obj.genus

    formatted_genus.short_description = 'Genus'

    def formatted_species(self, obj):
        """
        Format the species column to be more reader friendly
        """
        species = " ||| "
        for species_class in obj.species:
            name, confidence = species_class
            species += f"{name}: {confidence:.2f}% ||| "

        return species

    formatted_species.short_description = 'Species'

    def thumbnail(self, obj):
        if obj.frontal_image and obj.frontal_image.image:
            return format_html('<img src="{}" width="60" />', obj.frontal_image.image.url)
        return "No image"

    thumbnail.short_description = 'Thumbnail'

    def display_all_images(self, obj):
        """
        Adds ability for admin to view the images themselves in the specimenupload table
        """
        html = ""
        if obj.frontal_image and obj.frontal_image.image:
            html += mark_safe(f'''
                <div style="display: inline-block; margin-right: 10px; text-align: center;">
                    <strong>Frontal</strong><br>
                    <img src="{obj.frontal_image.image.url}" width="150" />
                </div>
            ''')
        if obj.dorsal_image and obj.dorsal_image.image:
            html += mark_safe(f'''
                <div style="display: inline-block; margin-right: 10px; text-align: center;">
                    <strong>Dorsal</strong><br>
                    <img src="{obj.dorsal_image.image.url}" width="150" />
                </div>
            ''')
        if obj.caudal_image and obj.caudal_image.image:
            html += mark_safe(f'''
                <div style="display: inline-block; margin-right: 10px; text-align: center;">
                    <strong>Caudal</strong><br>
                    <img src="{obj.caudal_image.image.url}" width="150" />
                </div>
            ''')
        if obj.lateral_image and obj.lateral_image.image:
            html += mark_safe(f'''
                <div style="display: inline-block; margin-right: 10px; text-align: center;">
                    <strong>Lateral</strong><br>
                    <img src="{obj.lateral_image.image.url}" width="150" />
                </div>
            ''')
        if not html:
            return "No images available."
        return mark_safe(html)
    display_all_images.short_description = 'Specimen Images'


@admin.register(ValidClasses)
class ValidClassesAdmin(admin.ModelAdmin):
    def delete_queryset(self, request, queryset):
        for obj in queryset:
            obj.delete()
