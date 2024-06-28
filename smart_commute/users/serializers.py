from rest_framework import serializers

from .models import Employee, ModelCombination, Shift, ModelWiseEmployeeRoute, Vehicle, ModelResultVehicleWise
import json


class EmployeeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Employee):
            return {
                'name': o.name,
                'id': o.id
            }
        return super().default(o)


def employee_to_dict(o):
    if isinstance(o, Employee):
        return {
            'name': o.name,
            'id': o.id
        }
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')


class ShiftSerializer(serializers.ModelSerializer):
    class Meta:
        model = Shift
        fields = ['id', 'shift_id']


class VehicleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vehicle
        fields = ['id', 'path_data', 'path_data_drop']


class ModelCombinationSerializer(serializers.ModelSerializer):
    shift = ShiftSerializer()

    class Meta:
        model = ModelCombination
        fields = '__all__'


class ModelVehicleSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelResultVehicleWise
        fields = '__all__'


class ModelCombinationDataSerializerRoute(serializers.ModelSerializer):
    class Meta:
        model = ModelWiseEmployeeRoute
        fields = ['id', 'vehicle_id']


class ModelCombinationDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelWiseEmployeeRoute
        fields = ['id', 'vehicle_id', 'sequence_no', 'distance', 'time']
