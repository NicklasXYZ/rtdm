# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
# ------------------------------------------------------------------------------#
#
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import uuid

from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.auth.models import PermissionsMixin
# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django std. lib.               #
# ------------------------------------------------------------------------------#
from django.db import models

# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from .managers import CustomBaseUserManager, DataPointManager, TrackManager

# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#


# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django extensions              #
# ------------------------------------------------------------------------------#


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class CustomBaseUser(AbstractBaseUser, PermissionsMixin):
    uid = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier of the user.",
        verbose_name="user uid",
    )  # A unique identifier of a "CustomBaseUser".
    email = models.EmailField(
        max_length=250,
        editable=True,
        null=False,
        blank=False,
        unique=True,
        help_text="The email of the user.",
        verbose_name="email",
    )  # A unique identifier of a "CustomBaseUser".
    password = models.CharField(  # Should be filled out by the "CustomBaseUser".
        max_length=250,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The password of the user.",
        verbose_name="password",
    )
    date_joined = models.DateTimeField(
        auto_now=False,
        auto_now_add=True,  # Set a date by default
        editable=False,  # Should not be validated or changeable!
        null=False,
        blank=False,
        unique=False,
        help_text="The date the user joined.",
        verbose_name="join date",
    )
    is_admin = models.BooleanField(
        default=False,
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="Specifies whether the user is an administrator.",
        verbose_name="is admin",
    )  # Specify whether the "CustomBaseUser" is an administrator or not.
    is_superuser = models.BooleanField(
        default=False,
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="Specifies whether the user is a superuser.",
        verbose_name="is superuser",
    )  # Specify whether the "CustomBaseUser" has all possible permissions.
    is_staff = models.BooleanField(
        default=False,
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="Specifies whether the user is a part of the staff.",
        verbose_name="is staff",
    )  # Specify whether the "CustomBaseUser" has certain permissions.

    EMAIL_FIELD = "email"
    USERNAME_FIELD = "email"  # Use the email as a unique identifier.
    REQUIRED_FIELDS = []  # Email and password are required by default.
    # Other required fields should be set here and
    # defined in the methods in the "managers.py" file.

    objects = CustomBaseUserManager()

    def __str__(self):
        """"""
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        """"""
        self.full_clean()
        super().save(*args, **kwargs)

    class Meta:
        """Metadata belonging to a "CustomBaseUser"."""

        verbose_name = "Custom Base User"
        verbose_name_plural = "Custom Base Users"
        abstract = (
            False  # The "CustomBaseUser" class is not an abstract base class.
        )
        ordering = ["date_joined"]


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class DataPoint(models.Model):
    uid = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier for the datapoint.",
        verbose_name="datapoint uid",
    )  # A unique identifier of a "DataPoint".
    internal_timestamp = models.DateTimeField(
        auto_now=False,
        auto_now_add=True,
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The internally set timestamp.",
        verbose_name="internal timestamp",
    )
    external_timestamp = models.DateTimeField(
        auto_now=False,
        auto_now_add=False,
        editable=True,
        null=True,
        blank=False,
        unique=False,
        help_text="The externally given timestamp.",
        verbose_name="external timestamp",
    )
    latitude = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The latitude of the location in degrees.",
        verbose_name="latitude",
    )
    longitude = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The longitude of the location in degrees.",
        verbose_name="longitude",
    )

    objects = DataPointManager()

    def __str__(self):
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Data Point"
        verbose_name_plural = "Data Points"
        abstract = False  # The "DataPoint" class is not an abstract base class!
        ordering = ["external_timestamp"]


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class Track(models.Model):
    uid = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier for the datapoint.",
        verbose_name="datapoint uid",
    )  # A unique identifier of a "Track".
    start_timestamp = models.DateTimeField(
        auto_now=False,
        auto_now_add=False,
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="Start timestamp.",
        verbose_name="start timestamp",
    )
    end_timestamp = models.DateTimeField(
        auto_now=False,
        auto_now_add=False,
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="End timestamp.",
        verbose_name="end timestamp",
    )
    min_latitude = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The min latitude of the location in degrees.",
        verbose_name="max latitude",
    )
    max_latitude = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The max latitude of the location in degrees.",
        verbose_name="max latitude",
    )
    min_longitude = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The min longitude of the location in degrees.",
        verbose_name="min longitude",
    )
    max_longitude = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The max longitude of the location in degrees.",
        verbose_name="max longitude",
    )
    duration = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The duration of the track in seconds.",
        verbose_name="duration",
    )
    eulidean_length = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The total length of the track in meters using the "
        + "euclidean distance.",
        verbose_name="length",
    )
    manhatten_length = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The total length of the track in meters using the "
        + "manhatten distance.",
        verbose_name="length",
    )

    objects = TrackManager()

    def __str__(self):
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Track"
        verbose_name_plural = "Tracks"
        abstract = False  # The "Track" class is not an abstract base class!
        ordering = ["start_timestamp"]
