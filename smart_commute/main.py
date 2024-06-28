import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smart_commute.settings')
django.setup()

from users.models import Employee
from users.algorithms import k_means_algorithm, clarke_wright_savings_complete


# Define the main function
def main():
    employees = Employee.objects.all().order_by('id')
    employee_coordinates = Employee.objects.values_list('coordinates', flat=True).order_by('id')
    employee_ids = Employee.objects.values_list('id', flat=True).order_by('id')
    employee_codes = Employee.objects.values_list('employee_code', flat=True).order_by('id')

    # Initialize the list with the first coordinate
    final_list_coord = [[24.834928, 67.37418]]

    for coord in employee_coordinates:
        coord_arr = coord.split(', ')
        list_coord = [float(coord_arr[0]), float(coord_arr[1])]
        final_list_coord.append(list_coord)

    # Call the k-means algorithm with the list of coordinates
    # k_means_algorithm(final_list_coord)
    clarke_wright_savings_complete(final_list_coord, 'EC', 'pick', 8, 10)


# Call the main function
if __name__ == '__main__':
    main()
