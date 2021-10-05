import json
from typing import Any, Union

import msgpack
from django.db import models

# Helper methods


def load(message: Union[None, bytes]) -> Union[None, Any]:
    """
    Convert the given input from bytes to Python native types.

    Args:
        message (Union[None, bytes]): A message that should converted to a
            Python native type.

    Raises:
        TypeError: If the input is not of type 'bytes' or 'None'.

    Returns:
        Union[None, Any]: The decoded input message as a Python native type or
            None.
    """
    if message is not None:
        if isinstance(message, bytes):
            data = msgpack.loads(message)
        else:
            error = (
                "The given value has the wrong type! "
                + "The given input needs to be of type 'bytes', but "
                + f"is currently of type {type(message)}"
            )
            raise TypeError(error)
    else:
        data = None
    return data


def dump(message: Union[None, Any]) -> bytes:
    """
    Convert given input to bytes: Any --> bytes.

    Args:
        message (Union[None, bytes]): A message that should be converted to \
            bytes.

    Raises:
        TypeError: If the input is not of type 'bytes' or 'None'.

    Returns:
        bytes: The input message as a bytes.
    """
    return msgpack.dumps(message)


# New model fields


class MsgPackField(models.Field):
    def __init__(self, *args, **kwargs):
        kwargs["blank"] = True
        super().__init__(*args, **kwargs)

    def get_db_prep_value(self, value, connection=None, prepared=False):
        if not prepared:
            value = dump(value)
        if value is not None:
            return connection.Database.Binary(value)
        return None

    def get_default(self):
        if self.has_default() and not callable(self.default):
            return self.default
        default = super().get_default()
        if default == "":
            return None
        return default

    def to_python(self, value):
        return load(value)

    def from_db_value(self, value, *args):
        return self.to_python(value)

    def get_prep_value(self, value):
        return self.to_python(value)

    def value_to_string(self, obj):
        if hasattr(self, "_get_val_from_obj"):
            value = self._get_val_from_obj(obj)
        else:
            value = super().value_from_object(obj)
        return json.dumps(value, indent="\t", separators=(", ", ": "))

    def get_internal_type(self):
        return "BinaryField"
