from django.contrib.auth.models import AbstractUser
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin, Permission, Group


class EmployeeUser(AbstractUser):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    username = models.CharField(max_length=30, unique=True)
    email = models.EmailField(unique=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    user_permissions = models.ManyToManyField(
        Permission,
        verbose_name='user permissions',
        blank=True,
        related_name='employee_user_permissions',  # Add this line
        help_text='Specific permissions for this user.'
    )

    groups = models.ManyToManyField(
        Group,
        verbose_name='groups',
        blank=True,
        related_name='employee_user_groups',  # Add this line
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.'
    )

    def __str__(self):
        return self.username


class AdminUser(AbstractUser):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    username = models.CharField(max_length=30, unique=True)
    email = models.EmailField(unique=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=True)

    user_permissions = models.ManyToManyField(
        Permission,
        verbose_name='user permissions',
        blank=True,
        related_name='admin_user_permissions',  # Add this line
        help_text='Specific permissions for this user.'
    )
    groups = models.ManyToManyField(
        Group,
        verbose_name='groups',
        blank=True,
        related_name='admin_user_groups',  # Add this line
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.'
    )

    def __str__(self):
        return self.username


class Department(models.Model):
    dept_id = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return f' {self.dept_id} - {self.name}'


class Location(models.Model):
    location_id = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return f' {self.location_id} - {self.name}'


class Shift(models.Model):
    shift_id = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, unique=True)
    slab = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return f' {self.shift_id} - {self.name}'


class JobTitle(models.Model):
    job_id = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return f' {self.job_id} - {self.name}'


class Employee(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    user = models.OneToOneField(EmployeeUser, on_delete=models.CASCADE, related_name='employee_profile')
    employee_code = models.CharField(max_length=10, unique=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30, blank=True, null=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='M')
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    job_title = models.ForeignKey(JobTitle, on_delete=models.CASCADE)
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    address = models.CharField(max_length=200, blank=True, null=True)
    address1 = models.CharField(max_length=255, blank=True, null=True)
    address2 = models.CharField(max_length=255, blank=True, null=True)  # Optional field
    city = models.CharField(max_length=100, blank=True, null=True)
    state = models.CharField(max_length=100, blank=True, null=True)
    postcode = models.CharField(max_length=20, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    contact_no = models.CharField(max_length=100)
    shift = models.ForeignKey(Shift, on_delete=models.CASCADE)
    coordinates = models.CharField(max_length=50)
    is_staff = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.employee_code}:    {self.first_name} {self.last_name}'

    def to_dict(self):
        return {
            'id': self.id,
            'employee_code': self.employee_code,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'coordinates': self.coordinates,
        }


class EmployeeFromFile(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    employee_group = models.CharField(max_length=100)
    department = models.CharField(max_length=100)
    address_line_1 = models.CharField(max_length=255, blank=True, null=True)
    address_line_2 = models.CharField(max_length=255, blank=True, null=True)
    address_line_3 = models.CharField(max_length=255, blank=True, null=True)
    contact_number = models.CharField(max_length=15)
    shift = models.CharField(max_length=50)
    coordinates = models.CharField(max_length=50)

    def __str__(self):
        return self.name


class Admin(models.Model):
    user = models.OneToOneField(AdminUser, on_delete=models.CASCADE, related_name='admin_profile')
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    is_staff = models.BooleanField(default=True)

    def __str__(self):
        return self.first_name


class Driver(models.Model):
    name = models.CharField(max_length=100)
    contact_number = models.CharField(max_length=15)
    license_number = models.CharField(max_length=50, unique=True)
    address = models.TextField(blank=True, null=True)
    date_of_birth = models.DateField(blank=True, null=True)
    hire_date = models.DateField()
    status = models.CharField(max_length=30, choices=[('active', 'Active'), ('inactive', 'Inactive')], default='active')

    def __str__(self):
        return self.name


class Vehicle(models.Model):
    registration_no = models.CharField(max_length=50, unique=True)
    chasis_no = models.CharField(max_length=50, unique=True)
    # vin = models.CharField(max_length=17, unique=True)  # VINs are usually 17 characters long
    color = models.CharField(max_length=30)
    type = models.CharField(max_length=30)
    model = models.CharField(max_length=50)
    brand = models.CharField(max_length=50)
    manufacture_date = models.DateField()
    registration_date = models.DateField()
    capacity = models.PositiveIntegerField()
    fuel_type = models.CharField(max_length=30)
    mileage = models.PositiveIntegerField()
    insurance_policy_no = models.CharField(max_length=50, unique=True, blank=True, null=True)
    insurance_expiry_date = models.DateField(blank=True, null=True)
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE, blank=True, null=True)
    path_data = models.JSONField(blank=True, null=True)
    path_data_drop = models.JSONField(blank=True, null=True)
    status = models.CharField(max_length=30,
                              choices=[('active', 'Active'), ('sold', 'Sold'), ('out_of_service', 'Out of Service')],
                              default='active')

    def __str__(self):
        return f"{self.brand} {self.model} ({self.registration_no})"


class OptimizePaths(models.Model):
    date = models.DateField()
    vehicle_no = models.IntegerField()
    distance = models.FloatField()
    time = models.FloatField()
    path_data = models.JSONField()

    def __str__(self):
        return f"{self.vehicle_no} : {self.path_data}"


class DriverVehicleMapping(models.Model):
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE)
    assignment_date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.driver.name} - {self.vehicle.registration_no}"


class ChangeRequest(models.Model):
    REQUEST_TYPE_CHOICES = [
        ('address', 'Address'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('denied', 'Denied'),
    ]

    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    request_type = models.CharField(max_length=20, choices=REQUEST_TYPE_CHOICES)
    old_value = models.CharField(max_length=255)
    new_value = models.CharField(max_length=255)
    reason = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.employee.name} - {self.get_request_type_display()} change request'


class ModelCombination(models.Model):
    ROUTE_TYPE_CHOICES = [
        ('pick', 'pick-up'),
        ('drop', 'drop_off'),
    ]
    no_of_vehicles = models.IntegerField()
    shift = models.ForeignKey(Shift, on_delete=models.CASCADE)
    type_id = models.CharField(max_length=20)
    first_level_algo = models.CharField(max_length=255, choices=ROUTE_TYPE_CHOICES)
    second_level_algo = models.CharField(max_length=255, blank=True, null=True)
    analysis = models.TextField(blank=True, null=True)
    is_implemented = models.BooleanField(blank=True, null=True)

    def __str__(self):
        return f"{self.shift.shift_id} - {self.type_id} - {self.first_level_algo} - {self.second_level_algo}"


class ModelWiseEmployeeRoute(models.Model):
    model_comb = models.ForeignKey(ModelCombination, on_delete=models.CASCADE)
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    sequence_no = models.IntegerField()
    distance = models.FloatField()
    time = models.FloatField()
    URL = models.TextField(blank=True)

    def __str__(self):
        return f"Route for Vehicle {self.vehicle.id}, Employee {self.employee.employee_code}, Sequence {self.sequence_no}"


class ModelResultVehicleWise(models.Model):
    model_comb = models.ForeignKey(ModelCombination, on_delete=models.CASCADE)
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE)
    path_data = models.JSONField(blank=True, null=True)

    def __str__(self):
        return f"{self.model_comb}"


class EmployeeVehicleMapping(models.Model):
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.vehicle.registration_no} - {self.employee.name}"
