# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
# ------------------------------------------------------------------------------#
# None
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import logging

from django.apps import apps
# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django std. lib.               #
# ------------------------------------------------------------------------------#
from django.conf import settings
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
# from asgiref.sync import sync_to_async # Use database_sync_to_async
from redisxchange.xchanges import RedisQueueMessageExchange
# from django.contrib.auth import get_user_model
# from django.shortcuts import get_object_or_404
# from django.core import serializers
# from django.utils import timezone
# from django.db import models
# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django extensions              #
# ------------------------------------------------------------------------------#
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .helpers.DataPoint import DataPoint
from .helpers.DataPointStream import DataPointStream
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from .serializers import CustomBaseUserSerializer, DataPointSerializer
from .utils import parse_date

# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
logging.basicConfig(level=logging.DEBUG)
exchange = RedisQueueMessageExchange()
dps = DataPointStream(min_time=600, min_distance=300,)


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class UserView(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    mixins.DestroyModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    # --------------------------------------------------------------------------#
    # View attributes                                                          #
    # --------------------------------------------------------------------------#
    model = apps.get_model(settings.APP_NAME, "CustomBaseUser")
    queryset = model.objects.all()
    serializer_class = CustomBaseUserSerializer
    pk_url_kwarg = "pk"
    lookup_field = "pk"

    def get_instance(self):
        """"""
        return self.request.user

    def get_permissions(self):
        """"""
        if self.action == "create":
            self.permission_classes = [AllowAny]
        return super().get_permissions()

    @action(["get", "put", "patch", "delete", "options"], detail=False)
    def me(self, request, *args, **kwargs):
        self.get_object = self.get_instance
        if request.method == "GET":
            return self.retrieve(request, *args, **kwargs)
        elif request.method == "PUT":
            return self.update(request, *args, **kwargs)
        elif request.method == "PATCH":
            return self.partial_update(request, *args, **kwargs)
        elif request.method == "DELETE":
            return self.destroy(request, *args, **kwargs)
        elif request.method == "OPTIONS":
            meta = self.metadata_class()
            data = meta.determine_metadata(request, self)
            return Response(data=data, status=status.HTTP_200_OK)


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class DataPointView(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    # --------------------------------------------------------------------------#
    # View attributes                                                          #
    # --------------------------------------------------------------------------#
    model = apps.get_model(settings.APP_NAME, "DataPoint")
    # Lookup options
    lookup_field = "pk"  # Use the primary_key (pk) for lookups

    # --------------------------------------------------------------------------#
    # Methods                                                                  #
    # --------------------------------------------------------------------------#
    def create(self, request, *args, **kwargs):
        datapoint_serializer = DataPointSerializer(data=request.data,)
        datapoint_serializer.is_valid(raise_exception=True)
        # datapoint_serializer.save(
        #     user = str(request.user.uid),
        # )
        datapoint_serializer.save()
        return_data = getattr(datapoint_serializer, "data")
        # TODO:
        # # Put data in FIFO queue
        # response_data = exchange.publish(
        #     message = return_data,
        #     name = "tracking",
        # )
        # # Process the data
        # DataPointStream(
        #     min_time = 600,
        #     min_distance = 300,
        # ).update()
        # TEMP
        time = parse_date(return_data["external_timestamp"])
        dps.update(
            DataPoint(
                latitude=return_data["latitude"],
                longitude=return_data["longitude"],
                time=time,
            )
        )
        response = Response(data=return_data, status=status.HTTP_201_CREATED,)
        return response

    def retrieve(self, request, *args, **kwargs):
        try:
            queryset = self.model.objects.get(uid=kwargs["pk"],)
            datapoint_serializer = DataPointSerializer(
                queryset, many=False, read_only=True,
            )
            return_data = getattr(datapoint_serializer, "data")
            response = Response(data=return_data, status=status.HTTP_200_OK,)
        except self.model.DoesNotExist:
            response = Response(status=status.HTTP_404_NOT_FOUND,)
        return response

    def list(self, request, *args, **kwargs):
        queryset = self.model.objects.filter(user=str(request.user.uid),)
        datapoint_serializer = DataPointSerializer(
            queryset, many=True, read_only=True,
        )
        return_data = getattr(datapoint_serializer, "data")
        response = Response(data=return_data, status=status.HTTP_200_OK,)
        return response
