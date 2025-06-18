from django.contrib import admin
from django.urls import path
from django.utils.html import format_html
from django.utils.safestring import mark_safe

# Register your models here.
from .models import SpecimenUpload, Image, KnownSpecies, Genus, TrainingDatabase, User, ValidClasses

admin.site.register(KnownSpecies)
admin.site.register(Genus)
admin.site.register(TrainingDatabase)
admin.site.register(User)


@admin.register(SpecimenUpload)
class SpecimenUploadAdmin(admin.ModelAdmin):
    """
    Add formatting for the specimen upload view on the admin page
    """
    list_display = ('id', 'formatted_genus', 'formatted_species', 'final_identification', 'display_all_images')
    list_filter = ('final_identification', )
    readonly_fields = ['display_all_images', 'formatted_genus', 'formatted_species']
    fields = ('display_all_images', 'formatted_genus', 'formatted_species', 'final_identification')

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
