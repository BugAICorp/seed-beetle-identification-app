import os
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.core.exceptions import ValidationError
from django.core.files.storage import default_storage
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _
from django.core.files.base import ContentFile
from PIL import Image as PILImage
from io import BytesIO


class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.is_usda = email.lower().strip().endswith('@usda.gov')  # this needs to change to @usda.gov
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('admin', True)
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_active', True)
        return self.create_user(email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    user_id = models.AutoField(primary_key=True)
    email = models.EmailField(unique=True, max_length=255)
    name = models.CharField(max_length=255, null=True, blank=True)
    is_email_verified = models.BooleanField(default=False)
    is_active = models.BooleanField(default=False)
    is_usda = models.BooleanField(default=False)
    admin = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='custom_user_set',  # Custom related_name to avoid clash
        blank=True
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='custom_user_permissions_set',  # Custom related_name to avoid clash
        blank=True
    )

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.email


def default_genus():
    return (None, 0.0)


def default_species():
    return [(None, 0.0)]


class SpecimenUpload(models.Model):
    id = models.AutoField(primary_key=True)  # Explicit primary key

    user = models.ForeignKey('port_inspector_app.User', on_delete=models.CASCADE, related_name="uploads")

    upload_date = models.DateTimeField(auto_now_add=True)

    frontal_image = models.ForeignKey('port_inspector_app.Image', on_delete=models.CASCADE, related_name="frontal_image", null=True, blank=True)
    dorsal_image = models.ForeignKey('port_inspector_app.Image', on_delete=models.CASCADE, related_name="dorsal_image", null=True, blank=True)
    caudal_image = models.ForeignKey('port_inspector_app.Image', on_delete=models.CASCADE, related_name="caudal_image", null=True, blank=True)
    lateral_image = models.ForeignKey('port_inspector_app.Image', on_delete=models.CASCADE, related_name="lateral_image", null=True, blank=True)

    genus = models.JSONField(default=default_genus)
    species = models.JSONField(default=default_species)

    final_identification = models.TextField()

    def clean(self):
        # Perform validation if we already have a pk and have been saved
        if self.id:
            num_images = self.images.count()
            if num_images < 1 or num_images > 4:
                raise ValidationError(f"A SpecimenUpload must have between 1 and 4 images. Found {num_images}.")

        # Validate genus format
        if len(self.genus) != 2:
            raise ValidationError("Genus must be a tuple containing (genus_id, confidence_level).")

        # Validate species format
        if not isinstance(self.species, list) or not (1 <= len(self.species) <= 5):
            raise ValidationError("Species must be a list of 1 to 5 (species_id, confidence_level) tuples.")

    # Delete images on SpecimenUpload delete
    def delete(self, *args, **kwargs):
        for image in self.images.all():
            if image:
                image.delete()
        super().delete(*args, **kwargs)

    def __str__(self):
        return f"SpecimenUpload #{self.id} by {self.user.email} on {self.upload_date}"


class Image(models.Model):
    id = models.AutoField(primary_key=True)  # Explicit primary key
    specimen_upload = models.ForeignKey('port_inspector_app.SpecimenUpload', on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if self.image and not kwargs.get('raw', False):
            img = PILImage.open(self.image)
            img = img.convert('RGB')

            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)

            self.image.save(self.image.name, ContentFile(buffer.read()), save=False)

        super().save(*args, **kwargs)

    # TODO: fix, make sure our image files get deleted w/SpecimenUpload
    def delete(self, *args, **kwargs):
        # Delete the associated image file
        if self.image and default_storage.exists(self.image.name):
            print(f"Deleted Image: {self.image.name}")
            default_storage.delete(self.image.name)
        super().delete(*args, **kwargs)

    def __str__(self):
        return f"Image #{self.id} for SpecimenUpload #{self.specimen_upload.id} uploaded at {self.uploaded_at}"


class KnownSpecies(models.Model):
    id_num = models.AutoField(primary_key=True)
    species_name = models.CharField(max_length=255, unique=True)
    resource_link = models.URLField(blank=True, null=True)

    class Meta:
        verbose_name_plural = "Known Species"

    def __str__(self):
        return self.species_name


class Genus(models.Model):
    id_num = models.AutoField(primary_key=True)
    genus_name = models.CharField(max_length=255, unique=True)
    resource_link = models.URLField(blank=True, null=True)

    class Meta:
        verbose_name_plural = "Genera"

    def __str__(self):
        return self.genus_name


class TrainingDatabase(models.Model):
    genus = models.CharField(max_length=32)
    species = models.CharField(max_length=80)
    uniqueid = models.CharField(max_length=100, unique=True)
    view = models.CharField(max_length=4)
    specimenid = models.CharField(max_length=255)
    image = models.BinaryField()

    class Meta:
        verbose_name_plural = "Training Database"
        verbose_name = "training entry"

    def __str__(self):
        return f"{self.specimenid} - {self.view}"


class ValidClasses(models.Model):
    genus = models.CharField(max_length=32)
    species = models.CharField(max_length=80)

    class Meta:
        verbose_name_plural = "Classes used for training"
        verbose_name = "allowed genus/species"

    def save(self, *args, **kwargs):
        from beetle_detection import species_eval
        super().save(*args, **kwargs)
        species_eval.refresh_database()

    def delete(self, *args, **kwargs):
        TrainingDatabase.objects.filter(species=self.species).delete()
        super().delete(*args, **kwargs)

    def __str__(self):
        return f"{self.genus} {self.species}"
