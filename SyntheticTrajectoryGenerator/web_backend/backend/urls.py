# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
# ------------------------------------------------------------------------------#
# Description: https://docs.djangoproject.com/en/3.0/topics/http/urls/          #
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#

from django.contrib import admin
# ------------------------------------------------------------------------------#
#                         Import third-party libraries: Django std. lib.       #
# ------------------------------------------------------------------------------#
from django.urls import include, path
from rest_framework.routers import DefaultRouter
# ------------------------------------------------------------------------------#
#                         Import third-party libraries: Django extensions      #
# ------------------------------------------------------------------------------#
# pip install drf-nested-routers
from rest_framework_nested import routers

# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from .views import DataPointView, UserView

# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# Main url router
router = DefaultRouter()

# Top-level endpoints
router.register(
    "users", UserView, basename="user",
)  # Overwrite the djoser /user/ endpoint
datapoint_router = routers.NestedDefaultRouter(router, "users", lookup="user")
# /datapoints/{datapoint_pk}/
datapoint_router.register(
    "datapoints", DataPointView, basename="datapoint",
)

# Compile all the endpoints
urlpatterns = [
    path("api/v1/admin/", admin.site.urls),
    path(
        "api/v1/", include(router.urls)
    ),  # Place these urls first to override!
    path("api/v1/", include(datapoint_router.urls)),
]
