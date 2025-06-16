from django.contrib import admin

# Register your models here.
from .models import SpecimenUpload
from .models import KnownSpecies
from .models import Genus

admin.site.register(SpecimenUpload)
admin.site.register(KnownSpecies)
admin.site.register(Genus)
