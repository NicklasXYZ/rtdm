import logging
import traceback

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from rest_framework import serializers
from rest_framework.utils import model_meta


class UserSerializerMixin(serializers.ModelSerializer):
    def create(self, validated_data):
        serializers.raise_errors_on_nested_writes(
            method_name="create",
            serializer=self,
            validated_data=validated_data,
        )
        model_class = self.Meta.model
        # Remove many-to-many relationships from validated_data.
        # They are not valid arguments to the default `.create()` method,
        # as they require that the instance has already been saved.
        info = model_meta.get_field_info(model_class)
        many_to_many = {}
        for field_name, relation_info in info.relations.items():
            if relation_info.to_many and (field_name in validated_data):
                many_to_many[field_name] = validated_data.pop(field_name)
        try:
            # Use a custom manager for creating users...
            instance = model_class._default_manager.create_user(validated_data)
        except TypeError:
            tb = traceback.format_exc()
            msg = (
                "Got a `TypeError` when calling `%s.%s.create()`. \
                This may be because you have a writable field on the \
                serializer class that is not a valid argument to \
                `%s.%s.create()`. You may need to make the field \
                read-only, or override the %s.create() method to handle \
                this correctly.\nOriginal exception was:\n %s"
                % (
                    model_class.__name__,
                    model_class._default_manager.name,
                    model_class.__name__,
                    model_class._default_manager.name,
                    self.__class__.__name__,
                    tb,
                )
            )
            raise TypeError(msg)
        # Save many-to-many relationships after the instance is created.
        if many_to_many:
            for field_name, value in many_to_many.items():
                field = getattr(instance, field_name)
                field.set(value)
        return instance


class CustomBaseUserSerializer(UserSerializerMixin):
    class Meta:
        """"""

        model = get_user_model()
        fields = [
            "uid",
            "username",
            "password",
        ]
        read_only_fields = [
            "uid",
        ]
        # Lookup field used by the corresponding "UserView"
        extra_kwargs = {"lookup_field": "pk"}


class StringListField(serializers.ListField):
    """Extended list serializer class."""

    child = serializers.CharField()


class DeviceMotionDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = apps.get_model(settings.APP_NAME, "DeviceMotionData")
        fields = [
            "uid",
            "internal_timestamp",
            "external_timestamp",
            "x",
            "y",
            "z",
        ]
        read_only_fields = [
            "uid",
            "internal_timestamp",
        ]
        extra_kwargs = {"lookup_field": "pk"}


class MagnetometerDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = apps.get_model(settings.APP_NAME, "MagnetometerData")
        fields = [
            "uid",
            "internal_timestamp",
            "x",
            "y",
            "z",
            "magnitude",
        ]
        read_only_fields = [
            "uid",
            "internal_timestamp",
        ]
        extra_kwargs = {"lookup_field": "pk"}


class DataPointSerializer(serializers.ModelSerializer):
    class Meta:
        model = apps.get_model(settings.APP_NAME, "DataPoint")
        fields = [
            "uid",
            "internal_timestamp",
            "external_timestamp",
            "longitude",
            "latitude",
            "speed",
            "accuracy",
            "user",
            # Extra data
            "devicemotion",
            "magnetometer",
        ]
        read_only_fields = [
            "uid",
            "internal_timestamp",
            "user",
        ]
        extra_kwargs = {"lookup_field": "pk"}

    devicemotion = DeviceMotionDataSerializer(
        many=False, read_only=False, allow_null=True, required=False,
    )
    magnetometer = MagnetometerDataSerializer(
        many=False, read_only=False, allow_null=True, required=False,
    )

    def create(self, validated_data):
        """Create and save a 'DataPoint' model object to the database."""
        devicemotion_model = apps.get_model(
            settings.APP_NAME, "DeviceMotionData"
        )
        devicemotion_data = validated_data.pop("devicemotion", None)
        if devicemotion_data is not None:
            devicemotion = devicemotion_model(**devicemotion_data)
        else:
            devicemotion = None
        magnetometer_model = apps.get_model(
            settings.APP_NAME, "MagnetometerData"
        )
        magnetometer_data = validated_data.pop("magnetometer", None)
        if magnetometer_data is not None:
            magnetometer = magnetometer_model(**magnetometer_data)
        else:
            magnetometer = None
        datapoint_model = apps.get_model(settings.APP_NAME, "DataPoint")
        try:
            # Keep track of the fields that should be updated on the datapoint
            # model instance
            update_fields = []
            with transaction.atomic():
                datapoint = datapoint_model.objects.create(**validated_data,)
                # If provided add additional data to datapoint instance and
                # save to db
                if devicemotion is not None:
                    devicemotion.save()
                    datapoint.devicemotion = devicemotion
                    update_fields.append("devicemotion")
                if magnetometer is not None:
                    magnetometer.save()
                    datapoint.magnetometer = magnetometer
                    update_fields.append("magnetometer")
                if devicemotion is not None or magnetometer is not None:
                    datapoint.save(update_fields=update_fields)
        except IntegrityError:
            # If we can not save the extra data then just save the geolocation
            # data
            datapoint = datapoint_model.objects.create(**validated_data,)
            logging.warn("DataPoint transaction failed!")
        return datapoint


class RemoveDataPointSerializer(serializers.Serializer):
    class Meta:
        fields = [
            "start_time",
        ]

    start_time = serializers.DateTimeField(
        help_text="The start of the time period, for which data points \
        should be deleted.",
    )
