from django.urls import include, path
from rest_framework import routers

from .views import UserView

# Main url router
router = routers.DefaultRouter()

# Top-level endpoints
router.register(
    "users", UserView, basename="user",
)

# Compile all the endpoints
urlpatterns = [
    path("api/", include(router.urls)),
]
