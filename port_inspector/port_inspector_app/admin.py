from django.contrib import admin

# Register your models here.
from .models import SpecimenUpload, KnownSpecies, Genus, TrainingDatabase

admin.site.register(TrainingDatabase)
admin.site.register(SpecimenUpload)
admin.site.register(KnownSpecies)
admin.site.register(Genus)
