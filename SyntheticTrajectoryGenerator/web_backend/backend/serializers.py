# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
# ------------------------------------------------------------------------------#
#
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import traceback

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
#
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
#
# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django std. lib.               #
# ------------------------------------------------------------------------------#
from rest_framework import serializers
from rest_framework.utils import model_meta

# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django extensions              #
# ------------------------------------------------------------------------------#
#


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class UserSerializerMixin(serializers.ModelSerializer):
    """"""

    # --------------------------------------------------------------------------#
    # Serializer attributes                                                    #
    # --------------------------------------------------------------------------#
    #
    # --------------------------------------------------------------------------#
    # Methods                                                                  #
    # --------------------------------------------------------------------------#
    def create(self, validated_data):
        """
        Description:
        Args:
        Returns:
        """
        serializers.raise_errors_on_nested_writes(
            "create", self, validated_data
        )
        ModelClass = self.Meta.model
        # Remove many-to-many relationships from validated_data.
        # They are not valid arguments to the default `.create()` method,
        # as they require that the instance has already been saved.
        info = model_meta.get_field_info(ModelClass)
        many_to_many = {}
        for field_name, relation_info in info.relations.items():
            if relation_info.to_many and (field_name in validated_data):
                many_to_many[field_name] = validated_data.pop(field_name)
        try:
            # Use a custom manager for creating users...
            instance = ModelClass._default_manager.create_user(validated_data)
        except TypeError:
            tb = traceback.format_exc()
            msg = (
                "Got a `TypeError` when calling `%s.%s.create()`. "
                "This may be because you have a writable field on the "
                "serializer class that is not a valid argument to "
                "`%s.%s.create()`. You may need to make the field "
                "read-only, or override the %s.create() method to handle "
                "this correctly.\nOriginal exception was:\n %s"
                % (
                    ModelClass.__name__,
                    ModelClass._default_manager.name,
                    ModelClass.__name__,
                    ModelClass._default_manager.name,
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


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class CustomBaseUserSerializer(UserSerializerMixin):
    """"""

    # --------------------------------------------------------------------------#
    # Serializer attributes                                                    #
    # --------------------------------------------------------------------------#
    # ...
    # --------------------------------------------------------------------------#
    # Methods                                                                  #
    # --------------------------------------------------------------------------#
    # ...
    # --------------------------------------------------------------------------#
    # Metadata                                                                 #
    # --------------------------------------------------------------------------#
    class Meta:
        """"""

        model = get_user_model()
        fields = [
            "uid",
            "email",
            "password",
        ]
        read_only_fields = [  # Non-writable fields
            "uid",
        ]
        extra_kwargs = {  # Lookup field used by the corresponding "UserView"
            "lookup_field": "pk"
        }


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class DataPointSerializer(serializers.ModelSerializer):
    class Meta:
        model = apps.get_model(settings.APP_NAME, "DataPoint")
        fields = [
            "uid",
            "internal_timestamp",
            "external_timestamp",
            "longitude",
            "latitude",
            # "user",
        ]
        read_only_fields = [
            "uid",
            "internal_timestamp",
            # "user",
        ]
        extra_kwargs = {"lookup_field": "pk"}


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class TrackSerializer(serializers.ModelSerializer):
    class Meta:
        model = apps.get_model(settings.APP_NAME, "Track")
        fields = [
            "uid",
            "start_timestamp",
            "end_timestamp",
            "min_longitude",
            "max_longitude",
            "min_latitude",
            "max_latitude",
        ]
        read_only_fields = [
            "uid",
        ]
        extra_kwargs = {"lookup_field": "pk"}
