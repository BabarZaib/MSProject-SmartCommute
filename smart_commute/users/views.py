import datetime
import json

import pandas as pd
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.db import transaction
from django.db.models import Count
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from .algorithms import k_means_algorithm, clarke_wright_savings_complete
from .forms import EmployeeRegistrationForm, AdminRegistrationForm, EmployeeUserCreationForm, CustomAuthenticationForm, \
    UploadFileForm, VehicleForm, DriverForm, ShiftForm, DriverSelectionForm, ChangeRequestForm
from .models import Employee, Department, JobTitle, Location, Shift, EmployeeUser, Driver, Vehicle, OptimizePaths, \
    DriverVehicleMapping, ChangeRequest, ModelCombination, ModelWiseEmployeeRoute, ModelResultVehicleWise
from .serializers import ModelCombinationSerializer, ModelCombinationDataSerializer, \
    ModelCombinationDataSerializerRoute, ModelVehicleSerializer

current_date = timezone.now()  # + datetime.timedelta(days=1)


def employee_registration(request):
    user_message = ''
    success_message = ''
    emp_id = 1

    if request.method == 'POST':
        if 'update' in request.POST:
            emp_id = request.POST.get('emp_id')
            employee_instance = Employee.objects.get(pk=emp_id)
            employee_form = EmployeeRegistrationForm(request.POST, instance=employee_instance)
            employee_form.save()
            return redirect('all_employees')  # Redirect to a success page
        user_form = EmployeeUserCreationForm(request.POST)
        employee_form = EmployeeRegistrationForm(request.POST)
        if user_form.is_valid() and employee_form.is_valid():
            user = user_form.save()
            employee = employee_form.save(commit=False)
            employee.user = user
            employee.save()
            success_message = "Employee Registered Successfully."
            return redirect('all_employees')  # Redirect to a success page
    else:
        user_form = EmployeeUserCreationForm()
        employee_form = EmployeeRegistrationForm()
        last_employee = Employee.objects.all().order_by('id').last()
        if last_employee:
            last_code = last_employee.employee_code
            last_number = int(last_code)
            emp_id = last_number + 1

    return render(request, 'users/employee_registration.html', {'user_form': user_form,
                                                                'employee_form': employee_form,
                                                                'user_message': user_message,
                                                                'emp_id': emp_id,

                                                                # 'carrier_message': carrier_form.
                                                                'success_message': success_message})


def edit_employee(request, emp_id):
    user_form = None
    employee_form = None
    if request.method == 'POST':
        if 'edit_emp' in request.POST:
            employee_instance = Employee.objects.get(pk=emp_id)
            employee_form = EmployeeRegistrationForm(instance=employee_instance)
            user = employee_instance.user
            user_form = EmployeeUserCreationForm(instance=user)

    return render(request, 'users/employee_registration.html', {'user_form': user_form,
                                                                'employee_form': employee_form, 'emp_id': emp_id,
                                                                'edit_employee': True})


def admin_registration(request):
    user_message = ''
    success_message = ''
    if request.method == 'POST':
        user_form = UserCreationForm(request.POST)
        admin_form = AdminRegistrationForm(request.POST)
        if user_form.is_valid() and admin_form.is_valid():
            user = user_form.save()
            admin = admin_form.save(commit=False)
            admin.user = user
            admin.save()
            success_message = "Admin Registered Successfully."
            return redirect('success_url')  # Redirect to a success page
    else:
        user_form = UserCreationForm()
        admin_form = AdminRegistrationForm()
    return render(request, 'users/admin_registration.html', {'user_form': user_form,
                                                             'admin': admin_form,
                                                             'user_message': user_message,
                                                             # 'carrier_message': carrier_form.
                                                             'success_message': success_message})


def dashboard(request):
    return render(request, 'users/dashboard.html')


def emp_dashboard(request):
    employee = get_object_or_404(Employee, user=request.user)
    return render(request, 'users/emp_dashboard.html', {'employee': employee})


def index(request):
    return render(request, 'users/index.html')


def all_employees(request):
    employees = Employee.objects.all()
    return render(request, 'users/all_employees.html', {'employees': employees})


def signin_view(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, request.POST)
        if form.is_valid():
            # Get the user credentials from the form
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            # Authenticate the user
            user = authenticate(request, username=username, password=password)

            if user is not None:
                # Log in the user
                login(request, user)
                messages.success(request, 'Login successful.')
                if user.is_superuser:
                    return redirect('dashboard')  # Replace with your superuser view name
                else:
                    return redirect('emp_dashboard')

            # return redirect('dashboard')
            else:
                messages.error(request, 'Invalid username or password.')

    else:
        form = CustomAuthenticationForm()

    return render(request, 'users/login.html', {'form': form, 'user_message': form.errors})


def handle_uploaded_file(f):
    df = pd.read_excel(f)
    df.fillna('', inplace=True)
    for _, row in df.iterrows():
        # Create a user for each employee
        email = f"{row['First Name'].replace(' ', '').lower()}{row['Last Name'].replace(' ', '').lower()}@example.com"  # Create a default email
        username = row['First Name'].replace(' ', '').lower() + row['Last Name'].replace(' ',
                                                                                         '').lower()  # Create a default username
        password = 'highRisk123'  # Use a default password or generate one

        # Check if the user already exists
        if not EmployeeUser.objects.filter(username=username).exists():
            user = EmployeeUser.objects.create_user(username=username, email=email, password=password)
        else:
            user = EmployeeUser.objects.get(username=username)

        ID = row['ID']
        # print(row)
        if not Employee.objects.filter(employee_code=ID).exists():
            first_name = row['First Name']
            last_name = row['Last Name']
            location = row['Location']
            location = get_object_or_404(Location, location_id=location)
            employee_group = row['Employee Group']
            job_title = get_object_or_404(JobTitle, job_id=employee_group)
            department = row['Department']
            department = get_object_or_404(Department, dept_id=department)
            address_line1 = row.get('Address Line 1', '')
            address_line2 = row.get('Address Line 2', '')
            address_line3 = row.get('Address Line 3', '')
            contact_no = row['Contact Number']
            shift = row['Shift']
            shift = get_object_or_404(Shift, shift_id=shift)
            coordinates = row['Coordinates']

            Employee.objects.create(
                user=user,
                employee_code=ID,
                first_name=first_name,
                last_name=last_name,
                department=department,
                job_title=job_title,
                location=location,
                address1=address_line1,
                address2=address_line2,
                contact_no=contact_no,
                shift=shift,
                coordinates=coordinates
            )
            print(f'id : {ID} , name :{first_name} {last_name}')


def create_model_combination(shift, route_type, first_level_algorithm, second_level_algorithm, analysis,
                             no_of_vehicles):
    with transaction.atomic():
        # Delete entries older than the delete_date
        ModelCombination.objects.filter(shift_id=shift, type_id=route_type,
                                        first_level_algo=first_level_algorithm,
                                        second_level_algo=second_level_algorithm).delete()

        # Create new entry after deleting old entries
        return ModelCombination.objects.create(shift_id=shift, type_id=route_type,
                                               first_level_algo=first_level_algorithm,
                                               second_level_algo=second_level_algorithm, analysis=analysis,
                                               no_of_vehicles=no_of_vehicles)


def create_modelwise_employee_route(model_comb, vehicle_id, emp_id, seq_no, distance, time):
    # Create new entry after deleting old entries
    return ModelWiseEmployeeRoute.objects.create(model_comb=model_comb, vehicle_id=vehicle_id,
                                                 employee_id=emp_id, sequence_no=seq_no, distance=distance, time=time)


def create_vehicle_model_wise(model_combination, vehicle, param):
    return ModelResultVehicleWise.objects.create(model_comb=model_combination, vehicle_id=vehicle, path_data=param)


def execute_model(request):
    shifts = Shift.objects.all()
    emp1 = Employee.objects.get(employee_code="1")
    emp2 = Employee.objects.get(employee_code="2")
    emp3 = Employee.objects.get(employee_code="3")

    data = {}
    data_list = []
    message = ''
    route_type = None
    final_route_dict = {}
    # model_execution goes here
    if request.POST:
        capacity = int(request.POST.get('capacity'))
        shift = request.POST.get('shift')
        route_type = request.POST.get('route_type')
        first_level_algorithm = request.POST.get('first_level_algorithm')
        second_level_algorithm = request.POST.get('second_level_algorithm')
        no_of_vehicles = int(request.POST.get('no_of_vehicles'))

        model_combination = create_model_combination(shift, route_type, first_level_algorithm, second_level_algorithm,
                                                     '', no_of_vehicles)

        try:
            data = {"vehicle": 1, "coordinates": [emp1.coordinates, emp2.coordinates, emp3.coordinates],
                    "employees": [emp1.id, emp2.id, emp3.id],
                    "distance": 20,
                    "time": "1 hour"}

            if shift:
                shift_object = Shift.objects.get(id=shift)
            else:
                shift_object = None

            employee_coordinates = Employee.objects.values_list('coordinates', flat=True).order_by('id')

            # Initialize the list with the first coordinate
            final_list_coord = [[24.834928, 67.37418]]

            for coord in employee_coordinates:
                coord_arr = coord.split(', ')
                list_coord = [float(coord_arr[0]), float(coord_arr[1])]
                final_list_coord.append(list_coord)

            ### Start saving results  ###

            if first_level_algorithm == 'kmeans':
                final_route_dict = k_means_call(final_list_coord)

            elif first_level_algorithm == 'clarke':
                final_route_dict = clarke_call(final_list_coord, shift_object.shift_id,
                                               route_type, capacity,
                                               no_of_vehicles)

            ### End saving results ###
            for final_route in final_route_dict:
                vehicle = int(final_route + 1)
                print(final_route_dict[final_route])
                employees = []
                employee_coordinates = []
                employee_list = final_route_dict[final_route]['route_vertex_index']
                create_vehicle_model_wise(model_combination, vehicle, final_route_dict[final_route])
                seq_no = 1
                idx = 0
                for emp_id in employee_list:
                    if emp_id != 0:
                        emp = Employee.objects.get(employee_code=emp_id)
                        employees.append(emp.employee_code)
                        employee_coordinates.append(emp.coordinates)
                        employee_distance = float(final_route_dict[final_route]['individual_distances'][idx])
                        employee_time = float(final_route_dict[final_route]['individual_durations'][idx])
                        create_modelwise_employee_route(model_combination, vehicle, emp.id, seq_no, employee_distance,
                                                        employee_time)
                        seq_no += 1
                        idx += 1

                data = {"vehicle": vehicle, "employees": employees, "coordinates": employee_coordinates,
                        "distance": str(final_route_dict[final_route]['distance']),
                        "time": str(final_route_dict[final_route]['duration'])}
                create_entries(vehicle, route_type, data)
                data_list.append(data)


        except Exception as e:
            print(f"An error occurred: {e}")
            message = 'Something went wrong while route optimization.'

    else:
        data = {"vehicle": 1, "coordinates": [emp1.coordinates, emp2.coordinates],
                "employees": [emp1.id, emp2.id],
                "distance": 20,
                "time": "1 hour"}
    return render(request, 'users/execute_model.html', {"shifts": shifts,
                                                        "data": data, "data_list": data_list, 'route_type': route_type,
                                                        'message': message})


@csrf_exempt
def execute_model_ajax(request):
    data = {}
    if request.POST:
        capacity = request.POST.get('capacity')
        no_of_vehicles = request.POST.get('no_of_vehicles')

        emp1 = Employee.objects.get(employee_code="1")
        emp2 = Employee.objects.get(employee_code="2")

        # employees_data = [emp.to_dict() for emp in [emp1, emp2]]
        employees_data = [emp1.coordinates, emp2.coordinates]

        data = {"vehicle": "1", "employees": [emp1.coordinates, emp2.coordinates], "distance": 20, "time": "1 hour"}
    return JsonResponse(data)


@csrf_exempt
def change_type_onchange(request):
    old_value = ''
    type = request.POST.get('type');
    try:
        employee = Employee.objects.get(user=request.user)
    except:
        employee = None
    if employee:
        if type == 'address':
            old_value = employee.address1
        elif type == 'shift':
            old_value = employee.shift.name
        else:
            old_value = 1

    data = {"old_value": old_value}
    return JsonResponse(data)


def bulk_employee_upload(request):
    if request.method == 'POST':
        upload_form = UploadFileForm(request.POST, request.FILES)

        if 'upload_emp' in request.POST and upload_form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            return redirect('all_employees')  # Redirect to a success page
    else:
        upload_form = UploadFileForm()

    return render(request, 'users/bulk_employee_upload.html', {'upload_form': upload_form})


def logout_view(request):
    logout(request)
    return redirect('login')


def vehicle_form(request):
    if request.method == 'POST':
        form = VehicleForm(request.POST)
        if 'update' in request.POST:
            vehicle_instance = Vehicle.objects.get(pk=request.POST.get('id'))
            form = VehicleForm(request.POST, instance=vehicle_instance)
        if form.is_valid():
            form.save()
            return redirect('vehicle_list')  # Replace with the name of your desired redirect view
    else:
        form = VehicleForm()

    return render(request, 'users/vehicle_form.html', {'form': form})


def driver_form(request):
    if request.method == 'POST':
        form = DriverForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('driver_list')  # Replace with the name of your desired redirect view
    else:
        form = DriverForm()

    return render(request, 'users/driver_form.html', {'form': form})


def shift_setup(request):
    shifts = Shift.objects.all()
    if request.method == 'POST':
        form = ShiftForm(request.POST)
        if 'update' in request.POST:
            shift_instance = Shift.objects.get(pk=request.POST.get('id'))
            form = ShiftForm(request.POST, instance=shift_instance)
        if form.is_valid():
            form.save()
            return redirect('shift_setup')  # Replace with the name of your desired redirect view
    else:
        form = ShiftForm()

    return render(request, 'users/shift_setup.html', {'form': form, "shifts": shifts})


def vehicle_list(request):
    vehicles = Vehicle.objects.all()
    if request.method == 'POST':
        if 'edit_vehicle' in request.POST:
            vehicle_id = request.POST.get('id')
            vehicle_instance = Vehicle.objects.get(pk=vehicle_id)
            form = VehicleForm(instance=vehicle_instance)
            return render(request, 'users/vehicle_form.html', {'form': form, 'edit_vehicle': True,
                                                               'vehicle_id': vehicle_id})
    else:
        return render(request, 'users/vehicle_list.html', {'vehicles': vehicles})


def driver_list(request):
    drivers = Driver.objects.all()
    return render(request, 'users/driver_list.html', {'drivers': drivers})


def model_wise_employee_route(request):
    model_combination = None
    model_combinations = ModelCombination.objects.all()
    model_wise_data = None  # ModelWiseEmployeeRoute.objects.all()
    model_grouped_data = ModelWiseEmployeeRoute.objects.values('model_comb_id').annotate(route_count=Count('id'))

    if request.method == 'POST':
        model_id = request.POST.get('model')
        if model_id:
            model_combination = ModelCombination.objects.get(id=model_id)
            model_wise_data = ModelWiseEmployeeRoute.objects.filter(model_comb_id=model_id)

    return render(request, 'users/model_wise_employee_route.html',
                  {'model_wise_data': model_wise_data,
                   'model_combinations': model_combinations, 'model_comb': model_combination})


def compare_models(request):
    model_combs = ModelCombination.objects.all()
    employee_list = Employee.objects.all()
    return render(request, 'users/compare_models.html',
                  {'model_combs': model_combs, 'employee_list': employee_list}
                  )


def manual_entry(request):
    vehicles = Vehicle.objects.all()
    employee_list = Employee.objects.all()
    return render(request, 'users/manual_entry.html',
                  {'vehicles': vehicles, 'employee_list': employee_list}
                  )


def model_data_vehicle_wise(request):

    vehicles = Vehicle.objects.all()
    model_combs = None
    if request.method == 'POST':
        selected_vehicle = request.POST.get('route')
        if selected_vehicle:
            model_combs = ModelResultVehicleWise.objects.filter(vehicle_id=selected_vehicle)

    return render(request, 'users/model_data_vehicle_wise.html',
                  {'vehicles': vehicles, 'model_combs': model_combs}
                  )


def fetch_model_vehicle_data(request):
    vehicle_id = request.GET.get('vehicle_id')
    model_id = request.GET.get('model_id')
    items = ModelResultVehicleWise.objects.get(model_comb=model_id, vehicle_id=vehicle_id)

    serializer = ModelVehicleSerializer(items)
    serialized_data = serializer.data

    return JsonResponse(serialized_data, safe=False)


def change_model(request):
    model_id = request.GET.get('model_id')
    items = ModelCombination.objects.exclude(id=model_id)
    model_data = ModelWiseEmployeeRoute.objects.filter(model_comb_id=model_id).values('vehicle_id').distinct()

    serializer = ModelCombinationSerializer(items, many=True)
    serialized_data = serializer.data

    return JsonResponse(serialized_data, safe=False)

    # return JsonResponse(list(items), safe=False)


def fetch_routes(request):
    model_id = request.GET.get('model_id')
    model_data = ModelWiseEmployeeRoute.objects.filter(model_comb_id=model_id).values('vehicle_id').distinct()

    serializer = ModelCombinationDataSerializerRoute(model_data, many=True)
    serialized_data = serializer.data

    return JsonResponse(serialized_data, safe=False)


def fetch_route_details(request):
    model1_id = request.GET.get('model1_id')
    model2_id = request.GET.get('model2_id')
    employee_id = request.GET.get('employee_id')
    model_data1 = ModelWiseEmployeeRoute.objects.get(model_comb_id=model1_id, employee_id=employee_id)
    model_data2 = ModelWiseEmployeeRoute.objects.get(model_comb_id=model2_id, employee_id=employee_id)

    model_list = []
    serializer = ModelCombinationDataSerializer(model_data1)
    serialized_data = serializer.data

    model_list.append(serialized_data)
    serializer = ModelCombinationDataSerializer(model_data2)
    serialized_data = serializer.data
    model_list.append(serialized_data)

    return JsonResponse(model_list, safe=False)


def employee_schedule(request):
    try:
        vehicles = Vehicle.objects.all()
        path_data = None

        employee = get_object_or_404(Employee, user=request.user)
        if employee:
            emp_id = employee.employee_code
            for vehicle in vehicles:
                if vehicle.path_data:
                    if emp_id in vehicle.path_data['employees']:
                        path_data = vehicle.path_data
                        print(path_data)
                    if path_data:
                        try:
                            driver_mapping = DriverVehicleMapping.objects.get(vehicle=vehicle)
                        except:
                            driver_mapping = None
                        driver = driver_mapping.driver if driver_mapping else None
    except:
        path_data = None
        driver = None
    return render(request, 'users/employee_schedule.html', {'path_data': path_data,
                                                            'current_date': current_date.date,
                                                            'driver': driver})


def create_entries(vehicle_number, route_type, listing):
    vehicle = Vehicle.objects.get(id=vehicle_number)
    if route_type == 'drop':
        vehicle.path_data_drop = listing
    else:
        vehicle.path_data = listing
    vehicle.save()


def driver_vehicle_mapping_view(request):
    if request.method == 'POST':
        vehicle_ids = request.POST.getlist('vehicle_id')
        driver_ids = request.POST.getlist('driver')

        for vehicle_id, driver_id in zip(vehicle_ids, driver_ids):
            vehicle = Vehicle.objects.get(id=vehicle_id)
            driver = Driver.objects.get(id=driver_id)
            # Create or update the mapping
            DriverVehicleMapping.objects.update_or_create(vehicle=vehicle, defaults={'driver': driver})

        return redirect('driver_vehicle_mapping')

    vehicles = Vehicle.objects.all()

    forms = []
    for vehicle in vehicles:
        try:
            drivermapping_set = DriverVehicleMapping.objects.get(vehicle=vehicle)
        except:
            drivermapping_set = None
        form = DriverSelectionForm(
            initial={
                'vehicle': vehicle,
                'driver': drivermapping_set.driver if drivermapping_set is not None else None})
        forms.append(form)
    vehicle_form_pairs = zip(vehicles, forms)
    context = {'vehicle_form_pairs': vehicle_form_pairs}
    return render(request, 'users/driver_vehicle_mapping.html', context)


def create_change_request(request):
    if request.method == 'POST':
        form = ChangeRequestForm(request.POST)
        if form.is_valid():
            change_request = form.save(commit=False)
            try:
                employee = Employee.objects.get(user=request.user)
            except:
                employee = None
            change_request.employee = employee  # Assuming Employee has a OneToOne relationship with User
            change_request.save()
            return redirect('change_request_list')
    else:
        form = ChangeRequestForm()
    return render(request, 'users/change_request_form.html', {'form': form})


def change_request_list(request):
    try:
        employee = Employee.objects.get(user=request.user)
    except:
        employee = None
    if employee:
        change_requests = ChangeRequest.objects.filter(employee=employee)
    else:
        change_requests = None
    return render(request, 'users/change_request_list.html',
                  {'change_requests': change_requests})


def k_means_call(final_list_coord):
    # Call the k-means algorithm with the list of coordinates
    final_routed_dict = k_means_algorithm(final_list_coord)
    print(final_routed_dict)
    return final_routed_dict


def clarke_call(final_list_coord, shift_id, route_type, max_capacity, no_of_vehicles):
    # Call the k-means algorithm with the list of coordinates
    final_routed_dict = clarke_wright_savings_complete(final_list_coord, shift_id, route_type, max_capacity,
                                                       no_of_vehicles)
    print(final_routed_dict)
    return final_routed_dict
