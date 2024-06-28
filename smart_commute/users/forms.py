from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

from .models import Employee, Admin, EmployeeUser, Vehicle, Driver, Shift, DriverVehicleMapping, ChangeRequest


class EmployeeRegistrationForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = ['employee_code', 'first_name', 'last_name', 'gender', 'department', 'job_title', 'location', 'shift',
                  'coordinates',
                  'address1', 'address2', 'state', 'postcode', 'city', 'country'
                  ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add any additional customization for form fields here
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({
                'placeholder': self.fields[field_name].label,
                'class': 'form-control',  # Add other classes as needed
            })


class AdminRegistrationForm(forms.ModelForm):
    class Meta:
        model = Admin
        fields = ['first_name', 'last_name']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add any additional customization for form fields here
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({
                'placeholder': self.fields[field_name].label,
                'class': 'form-control',  # Add other classes as needed
            })


class EmployeeUserCreationForm(UserCreationForm):
    class Meta:
        model = EmployeeUser
        fields = ['email', 'username', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name in self.fields:
            field = self.fields[field_name]
            field.widget.attrs.update({
                'class': 'form-control',
                'placeholder': field.label,
            })
            field.label = ''


class CustomAuthenticationForm(AuthenticationForm):
    # email = forms.EmailField(required=True, label="Enter Email Address..")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # del self.fields['username']
        for name in self.fields.keys():
            field = self.fields[name]
            self.fields[name].widget.attrs.update({
                'placeholder': self.fields[name].label,
                'class': 'form-control  form-control-user',
            })

            field.label = ''
            field.widget.attrs.update({'aria-label': field.label})


class UploadFileForm(forms.Form):
    file = forms.FileField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # del self.fields['username']
        for name in self.fields.keys():
            field = self.fields[name]
            self.fields[name].widget.attrs.update({
                'placeholder': self.fields[name].label,
                'class': 'form-control  file-upload-info',
            })


class VehicleForm(forms.ModelForm):
    class Meta:
        model = Vehicle
        fields = [
            'registration_no', 'chasis_no', 'color', 'type', 'model', 'brand',
            'manufacture_date', 'registration_date', 'capacity', 'fuel_type',
            'mileage', 'insurance_policy_no', 'insurance_expiry_date', 'status'
        ]
        widgets = {
            'manufacture_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control',
                                                       'placeholder': 'Select a date'}),
            'registration_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control',
                                                        'placeholder': 'Select a date'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # del self.fields['username']
        for name in self.fields.keys():
            field = self.fields[name]
            self.fields[name].widget.attrs.update({
                'placeholder': self.fields[name].label,
                'class': 'form-control  form-control-user',
            })


class DriverForm(forms.ModelForm):
    class Meta:
        model = Driver
        fields = ['name', 'contact_number', 'license_number', 'address', 'date_of_birth', 'hire_date', 'status']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # del self.fields['username']
        for name in self.fields.keys():
            field = self.fields[name]
            self.fields[name].widget.attrs.update({
                'placeholder': self.fields[name].label,
                'class': 'form-control  form-control-user',
            })


class ShiftForm(forms.ModelForm):
    class Meta:
        model = Shift
        fields = ["shift_id", "name", "slab"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # del self.fields['username']
        for name in self.fields.keys():
            field = self.fields[name]
            self.fields[name].widget.attrs.update({
                'placeholder': self.fields[name].label,
                'class': 'form-control  form-control-user',
            })

            field.label = ''
            field.widget.attrs.update({'aria-label': field.label})


class DriverSelectionForm(forms.ModelForm):
    driver = forms.ModelChoiceField(queryset=Driver.objects.all(), required=True)

    class Meta:
        model = DriverVehicleMapping
        fields = ['driver']

    def __init__(self, *args, **kwargs):
        vehicle = kwargs['initial']['vehicle']
        super().__init__(*args, **kwargs)
        # Get drivers who are already assigned to vehicles
        assigned_drivers = DriverVehicleMapping.objects.values_list('driver_id', flat=True)

        # Filter out assigned drivers, but include the current driver if this vehicle has one
        try:
            drivermappings = DriverVehicleMapping.objects.get(vehicle=vehicle)
        except:
            drivermappings = None

        if vehicle and drivermappings:
            current_driver = drivermappings.driver
            self.fields['driver'].queryset = Driver.objects.exclude(id__in=assigned_drivers).union(
                Driver.objects.filter(id=current_driver.id))
        else:
            self.fields['driver'].queryset = Driver.objects.exclude(id__in=assigned_drivers)

        # del self.fields['username']
        for name in self.fields.keys():
            field = self.fields[name]
            self.fields[name].widget.attrs.update({
                'placeholder': self.fields[name].label,
                'class': 'form-control  form-control-user',
            })


class ChangeRequestForm(forms.ModelForm):
    class Meta:
        model = ChangeRequest
        fields = ['request_type', 'old_value', 'new_value', 'reason']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # del self.fields['username']
        for name in self.fields.keys():
            field = self.fields[name]
            self.fields[name].widget.attrs.update({
                'placeholder': self.fields[name].label,
                'class': 'form-control  form-control-user',
            })
