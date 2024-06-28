from django.contrib.auth.backends import  BaseBackend
from .models import EmployeeUser, AdminUser


class EmployeeUserBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        # Implement authentication logic for ShipperUser
        try:
            user = EmployeeUser.objects.get(username=username)
            if user.check_password(password):
                return user
        except EmployeeUser.DoesNotExist:
            return None

    def get_user(self, user_id):
        try:
            return EmployeeUser.objects.get(pk=user_id)
        except EmployeeUser.DoesNotExist:
            return None


class AdminUserBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        # Implement authentication logic for CarrierUser
        try:
            user = AdminUser.objects.get(username=username)
            if user.check_password(password):
                return user
        except AdminUser.DoesNotExist:
            return None

    def get_user(self, user_id):
        try:
            return AdminUser.objects.get(pk=user_id)
        except AdminUser.DoesNotExist:
            return None