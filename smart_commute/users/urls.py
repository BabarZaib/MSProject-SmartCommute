from .views import dashboard, admin_registration, employee_registration, index, signin_view, logout_view, all_employees, \
    edit_employee, bulk_employee_upload, emp_dashboard, vehicle_form, driver_form, vehicle_list, driver_list, \
    execute_model, execute_model_ajax, shift_setup, employee_schedule, driver_vehicle_mapping_view, \
    create_change_request, change_request_list, change_type_onchange, model_wise_employee_route, compare_models, \
    change_model, fetch_routes, fetch_route_details, manual_entry, model_data_vehicle_wise, fetch_model_vehicle_data
from django.urls import path


urlpatterns = [
    path('', index, name='index'),
    path('login', signin_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('dashboard', dashboard, name='dashboard'),
    path('emp_dashboard', emp_dashboard, name='emp_dashboard'),
    path('employee_registration', employee_registration, name='employee_registration'),
    path('edit_employee/<int:emp_id>', edit_employee, name='edit_employee'),
    path('all_employees', all_employees, name='all_employees'),
    path('bulk_employee_upload', bulk_employee_upload, name='bulk_employee_upload'),
    path('admin_registration', admin_registration, name='admin_registration'),
    path('vehicle_form', vehicle_form, name='vehicle_form'),
    path('driver_form', driver_form, name='driver_form'),
    path('vehicle_list', vehicle_list, name='vehicle_list'),
    path('driver_list', driver_list, name='driver_list'),
    path('shift_setup', shift_setup, name='shift_setup'),
    path('execute_model', execute_model, name='execute_model'),
    path('employee_schedule', employee_schedule, name='employee_schedule'),
    path('driver_vehicle_mapping', driver_vehicle_mapping_view, name='driver_vehicle_mapping'),
    path('execute_model_ajax/', execute_model_ajax, name='execute_model_ajax'),
    path('change_model/', change_model, name='change_model'),
    path('fetch_routes/', fetch_routes, name='fetch_routes'),
    path('fetch_route_details/', fetch_route_details, name='fetch_route_details'),
    path('change_type_onchange/', change_type_onchange, name='change_type_onchange'),
    path('change-request/new/', create_change_request, name='create_change_request'),
    path('change-requests/', change_request_list, name='change_request_list'),
    path('model_wise_employee_route/', model_wise_employee_route, name='model_wise_employee_route'),
    path('compare_models/', compare_models, name='compare_models'),
    path('manual_entry/', manual_entry, name='manual_entry'),
    path('model_data_vehicle_wise/', model_data_vehicle_wise, name='model_data_vehicle_wise'),
    path('fetch_model_vehicle_data/', fetch_model_vehicle_data, name='fetch_model_vehicle_data'),
]