import logging
from collections import defaultdict
from typing import Any, Dict, List, Union

from django.apps import apps
from django.contrib.auth.base_user import BaseUserManager
from django.db import models


class BulkCreateManager:
    def __init__(self, min_chunk_size: int = 1, max_chunk_size: int = 2500):
        self._create_queues: Dict[str, List[Any]] = defaultdict(list)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_size = min_chunk_size
        self._check_vars()

    def set_chunk_size(self, size: int) -> None:
        if size >= self.max_chunk_size:
            self.chunk_size = self.max_chunk_size
        elif size <= self.min_chunk_size:
            self.chunk_size = self.min_chunk_size
        else:
            self.chunk_size = size

    def add(self, obj: Any) -> None:
        """Add an object to the model-specific queue."""
        model_class = type(obj)
        model_key = model_class._meta.label
        self._create_queues[model_key].append(obj)
        if len(self._create_queues[model_key]) >= self.chunk_size:
            logging.debug(
                "INFO: Inserting "
                + str(len(self._create_queues[model_key]))
                + " rows in the database.",
            )
            self._commit(model_class)

    def add_bulk(
        self, obj_list: List[Any], on_key: Union[None, str] = None
    ) -> None:
        """Add several objects to model-specific queues."""
        save_bulk = False
        for obj in obj_list:
            model_class = type(obj)
            model_key = model_class._meta.label
            self._create_queues[model_key].append(obj)
            if on_key is None:
                if len(self._create_queues[model_key]) >= self.chunk_size:
                    save_bulk = True
            else:
                if len(self._create_queues[on_key]) >= self.chunk_size:
                    save_bulk = True
        if save_bulk is True:
            for model_key in self._create_queues:
                model_class = apps.get_model(model_key)
                logging.debug(
                    "INFO: Inserting "
                    + str(len(self._create_queues[model_key]))
                    + " rows in the database. Model: "
                    + str(model_key),
                )
                self._commit(model_class)

    def done(self) -> None:
        """Always call this upon completion."""
        for model_name, model_objects in self._create_queues.items():
            if len(model_objects) > 0:
                logging.debug(
                    "INFO: Inserting "
                    + str(len(model_objects))
                    + " rows in the database.",
                )
            self._commit(apps.get_model(model_name))

    def _commit(self, model_class: Any) -> None:
        model_key = model_class._meta.label
        model_class.objects.bulk_create(self._create_queues[model_key])
        self._create_queues[model_key] = []

    def _check_vars(self) -> None:
        """Check and validate all class arguments on class instantiation."""
        if not isinstance(self.min_chunk_size, int):
            error = f"ARG 'min_chunk_size' is of type \
                {type(self.min_chunk_size)} but should be of type 'int'!"
            try:
                self.min_chunk_size = int(self.min_chunk_size)
            except Exception:
                raise TypeError(error)
        else:
            if self.min_chunk_size < 0:
                raise ValueError(
                    f"The given 'min_chunk_size' value is: \
                    {self.min_chunk_size}. It needs to be larger than 0."
                )
        if not isinstance(self.max_chunk_size, int):
            error = f"ARG 'max_chunk_size' is of type \
                {type(self.max_chunk_size)} but should be of type 'int'!"
            try:
                self.max_chunk_size = int(self.max_chunk_size)
            except Exception:
                raise TypeError(error)
        else:
            if self.max_chunk_size < 0:
                raise ValueError(
                    f"The given 'max_chunk_size' value is: \
                    {self.max_chunk_size}. It needs to be larger than 0."
                )


class UserManager(BaseUserManager):
    def create_user(self, username, password, **extra_fields):
        """Create and save a User with the given email and password."""
        username = username.strip()
        user = self.model(username=username, **extra_fields)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, username, password, **extra_fields):
        # These fields should be set to the following values
        # regardless of what is given as input
        extra_fields["is_admin"] = True
        extra_fields["is_superuser"] = True
        extra_fields["is_staff"] = True
        return self.create_user(username, password, **extra_fields)


class DataPointManager(models.Manager):
    pass


class TrajectoryManager(models.Manager):
    pass


class BreakPointManager(models.Manager):
    pass


class SequenceManager(models.Manager):
    pass
