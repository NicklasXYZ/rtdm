from typing import List

import ruuid  # pip install ruuid==0.3.0 --force --no-cach
from backend.fields import MsgPackField
from backend.managers import (
    BreakPointManager,
    DataPointManager,
    SequenceManager,
    TrajectoryManager,
    UserManager,
)
from backend.utils import color_hex
from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.auth.models import PermissionsMixin
from django.db import models


class User(AbstractBaseUser, PermissionsMixin):
    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"
        # The "User" class is not an abstract base class
        abstract = False
        ordering = ["date_joined"]

    USERNAME_FIELD = "username"
    REQUIRED_FIELDS: List[str] = []

    objects = UserManager()

    uid = models.UUIDField(
        primary_key=True,
        default=ruuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier of the user.",
        verbose_name="unique identifier",
    )
    username = models.CharField(
        max_length=250,
        editable=True,
        null=False,
        blank=False,
        unique=True,
        help_text="The username of the user.",
        verbose_name="username",
    )
    password = models.CharField(
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
    )  # Specify whether the "BaseUser" is an administrator or not.
    is_superuser = models.BooleanField(
        default=False,
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="Specifies whether the user is a superuser.",
        verbose_name="is superuser",
    )  # Specify whether the "BaseUser" has all possible permissions.
    is_staff = models.BooleanField(
        default=False,
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="Specifies whether the user is a part of the staff.",
        verbose_name="is staff",
    )  # Specify whether the "User" has certain permissions.

    def __str__(self):
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class Trajectory(models.Model):
    class Meta:
        verbose_name = "Trajectory"
        verbose_name_plural = "Trajectories"
        # The "Trajectory" class is not an abstract base class!
        abstract = False
        ordering = ["start_timestamp"]

    objects = TrajectoryManager()

    uid = models.UUIDField(
        primary_key=True,
        default=ruuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier of the trajectory.",
        verbose_name="unique identifier",
    )  # A unique identifier of a "Trajectory".
    color = models.CharField(
        default=f"#{color_hex()}",
        max_length=250,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The color used to represent the trajectory.",
        verbose_name="color",
    )
    start_timestamp = models.DateTimeField(
        auto_now=False,
        auto_now_add=False,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The first timestamp of a datapoint in the trajectory.",
        verbose_name="start timestamp",
    )
    end_timestamp = models.DateTimeField(
        auto_now=False,
        auto_now_add=False,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The last timestamp of a datapoint in the trajectory.",
        verbose_name="end timestamp",
    )
    user = models.UUIDField(
        editable=False,
        null=False,
        blank=False,
        unique=False,
        help_text="The unique identifier of the user.",
        verbose_name="user",
    )
    tag = models.CharField(
        max_length=250,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="A tag used to categorize trajectories.",
        verbose_name="tag",
    )
    trajectory = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="subtrajectories",
        help_text="The trajectory a subtrajectory belongs to.",
        verbose_name="trajectory",
    )

    def __str__(self):
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class DeviceMotionData(models.Model):
    class Meta:
        verbose_name = "Device Motion Data"
        verbose_name_plural = "Device Motion Data"
        # The "DeviceMotionData" class is not an abstract base class!
        abstract = False
        ordering = ["external_timestamp"]

    uid = models.UUIDField(
        primary_key=True,
        default=ruuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier for the device motion data.",
        verbose_name="device motion uid",
    )  # A unique identifier of the "DeviceMotionData".
    external_timestamp = models.DateTimeField(
        auto_now=False,
        auto_now_add=False,
        editable=True,
        null=False,  # Changed from True to False!
        blank=False,
        unique=False,
        help_text="The externally given timestamp.",
        verbose_name="external timestamp",
    )
    x = models.FloatField(  # noqa
        default=None,
        editable=True,
        null=True,  # TODO: False
        blank=True,  # TODO: False
        unique=False,
        help_text="",
        verbose_name="",
    )
    y = models.FloatField(  # noqa
        default=None,
        editable=True,
        null=True,  # TODO: False
        blank=True,  # TODO: False
        unique=False,
        help_text="",
        verbose_name="",
    )
    z = models.FloatField(  # noqa
        default=None,
        editable=True,
        null=True,  # TODO: False
        blank=True,  # TODO: False
        unique=False,
        help_text="",
        verbose_name="",
    )

    def __str__(self):
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class MagnetometerData(models.Model):
    class Meta:
        verbose_name = "Magnetometer Data"
        verbose_name_plural = "Magnetometer Data"
        # The "MagnetometerData" class is not an abstract base class!
        abstract = False
        ordering = ["internal_timestamp"]

    uid = models.UUIDField(
        primary_key=True,
        default=ruuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier for the magnetometer data.",
        verbose_name="magnetometer uid",
    )  # A unique identifier of the "MagnetometerData".
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
    x = models.FloatField(  # noqa
        default=None,
        editable=True,
        null=True,  # TODO: False
        blank=True,  # TODO: False
        unique=False,
        help_text="",
        verbose_name="",
    )
    y = models.FloatField(  # noqa
        default=None,
        editable=True,
        null=True,  # TODO: False
        blank=True,  # TODO: False
        unique=False,
        help_text="",
        verbose_name="",
    )
    z = models.FloatField(  # noqa
        default=None,
        editable=True,
        null=True,  # TODO: False
        blank=True,  # TODO: False
        unique=False,
        help_text="",
        verbose_name="",
    )
    magnitude = models.FloatField(
        default=None,
        editable=True,
        null=True,  # TODO: False
        blank=True,  # TODO: False
        unique=False,
        help_text="",
        verbose_name="",
    )

    def __str__(self):
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class DataPoint(models.Model):
    class Meta:
        verbose_name = "Data Point"
        verbose_name_plural = "Data Points"
        # The "DataPoint" class is not an abstract base class!
        abstract = False
        ordering = ["external_timestamp"]

    objects = DataPointManager()

    uid = models.UUIDField(
        primary_key=True,
        default=ruuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier of the datapoint.",
        verbose_name="datapoint uid",
    )  # A unique identifier of a "DataPoint".
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
    accuracy = models.FloatField(
        default=None,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The estimated horizontal accuracy of the location in \
            meters, if it is available.",
        verbose_name="accuracy",
    )
    speed = models.FloatField(
        default=None,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The speed in meters/second over ground, if it is available.",
        verbose_name="speed",
    )
    acceleration = models.FloatField(
        default=None,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The acceleration in meters/second^2 over ground, if it is \
        available.",
        verbose_name="speed",
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
    user = models.UUIDField(
        editable=False,
        null=False,
        blank=False,
        unique=False,
        help_text="The unique identifier of the user.",
        verbose_name="user",
    )
    label = models.CharField(
        max_length=250,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The label of the datapoint.",
        verbose_name="label",
    )
    weight = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The weight assigned to the datapoint.",
        verbose_name="weight",
    )
    dx = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The change in distance from previous datapoint.",
        verbose_name="dx",
    )
    dt = models.FloatField(
        editable=True,
        null=False,
        blank=False,
        unique=False,
        help_text="The change in time from the previous datapoint.",
        verbose_name="dt",
    )
    datapoints = MsgPackField(
        null=True,
        blank=True,
        help_text="A list of datapoints. The list is non-empty if the \
            datapoint is an aggregate datapoint representing several \
            datapoints that are in the vicinity of each other.",
        verbose_name="datapoints",
    )
    trajectory = models.ForeignKey(
        Trajectory,
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=False,
        unique=False,
        related_name="datapoints",
        help_text="The trajectory the datapoint is associated with.",
        verbose_name="trajectory",
    )

    # Extend datapoint model with extra 'devicemotion' data
    devicemotion = models.OneToOneField(
        DeviceMotionData,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        on_delete=models.CASCADE,
        related_name="datapoint_devicemotion",
        help_text="",
        verbose_name="",
    )

    # Extend datapoint model with extra 'magnetometer' data
    magnetometer = models.OneToOneField(
        MagnetometerData,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        on_delete=models.CASCADE,
        related_name="datapoint_magnetometer",
        help_text="",
        verbose_name="",
    )

    def __str__(self):
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class BreakPoint(models.Model):
    class Meta:
        verbose_name = "Break Point"
        verbose_name_plural = "Break Points"
        # The "BreakPoint" class is not an abstract base class!
        abstract = False
        ordering = ["scan_index"]

    objects = BreakPointManager()

    uid = models.UUIDField(
        primary_key=True,
        default=ruuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier of the breakpoint.",
        verbose_name="breakpoint uid",
    )  # A unique identifier of a "DataPoint".
    start_index = models.DateTimeField(
        auto_now=False,
        auto_now_add=False,
        editable=True,
        null=True,
        blank=False,
        unique=False,
        help_text="A timestamp.",
        verbose_name="",
    )
    scan_index = models.DateTimeField(
        auto_now=False,
        auto_now_add=False,
        editable=True,
        null=True,
        blank=False,
        unique=False,
        help_text="A timestamp.",
        verbose_name="",
    )
    last_index = models.DateTimeField(
        auto_now=False,
        auto_now_add=False,
        editable=True,
        null=True,
        blank=False,
        unique=False,
        help_text="A timestamp.",
        verbose_name="",
    )
    label = models.CharField(
        max_length=250,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text=".",
        verbose_name="",
    )
    user = models.UUIDField(
        editable=False,
        null=False,
        blank=False,
        unique=False,
        help_text="The unique identifier of the user.",
        verbose_name="user",
    )
    trajectory = models.ForeignKey(
        Trajectory,
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="breakpoints",
        help_text="The trajectory the breakpoint is associated with.",
        verbose_name="trajectory",
    )

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class Sequence(models.Model):
    class Meta:
        verbose_name = "Sequence"
        verbose_name_plural = "Sequences"
        abstract = False

    objects = SequenceManager()

    uid = models.UUIDField(
        primary_key=True,
        default=ruuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier of the sequence.",
        verbose_name="uid",
    )
    tokens = MsgPackField(
        null=True,
        blank=True,
        help_text="A list of tokens that make up a sequence.",
        verbose_name="tokens",
    )
    user = models.UUIDField(
        editable=False,
        null=False,
        blank=False,
        unique=False,
        help_text="The unique identifier of the user.",
        verbose_name="user",
    )

    def __str__(self):
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        self.full_clean(exclude=["tokens"])
        super().save(*args, **kwargs)


class SupportSet(models.Model):
    class Meta:
        verbose_name = "SupportSet"
        verbose_name_plural = "SupportSets"
        abstract = False
        ordering = ["updated"]

    uid = models.UUIDField(
        primary_key=True,
        default=ruuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="The unique identifier of the support set.",
        verbose_name="uid",
    )
    updated = models.DateTimeField(
        auto_now=True,
        editable=False,  # Should not be validated or changeable!
        null=False,
        blank=False,
        unique=False,
        help_text="The date the user joined.",
        verbose_name="join date",
    )
    patterns = MsgPackField(
        null=True,
        blank=True,
        help_text="A list of patterns that are (frequency, sequence) pairs.",
        verbose_name="patterns",
    )
    # Each user is associated with a single support set
    user = models.UUIDField(
        editable=False,
        null=False,
        blank=False,
        unique=False,
        help_text="The unique identifier of the user.",
        verbose_name="user",
    )

    def __str__(self):
        return f"{self.uid}"

    def save(self, *args, **kwargs):
        self.full_clean(exclude=["patterns"])
        super().save(*args, **kwargs)
