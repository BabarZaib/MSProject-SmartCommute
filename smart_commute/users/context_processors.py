from .models import ChangeRequest

def pending_requests_count(request):
    pending_count = ChangeRequest.objects.filter(status='pending').count()
    if pending_count == 0:
        pending_count = None
    return {'pending_count': pending_count}