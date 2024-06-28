from django.contrib import admin

from .models import Vehicle


# Register your models here.
@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    list_display = ('registration_no', 'chasis_no', 'brand', 'model', 'type', 'color',  'capacity', 'fuel_type', 'status')
    search_fields = ('registration_no', 'chasis_no', 'brand', 'model')
    list_filter = ('brand', 'type', 'fuel_type', 'status')