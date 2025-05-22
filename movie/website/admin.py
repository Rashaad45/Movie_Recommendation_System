from django.contrib import admin
from .models import Moviebase
from .models import upcoming

# Register your models here.

admin.site.register(Moviebase)
admin.site.register(upcoming)