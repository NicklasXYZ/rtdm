# import backend.DataStructures as dss

# boot_django.py
#
# to setup and configure, we use this file. Which is used by scripts
# that must run on a Django server as if it was running.
import os

import django
from django.conf import settings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))


def initialize_django():
    DEFAULT_APPS = [
        "django.contrib.admin",
        "django.contrib.auth",
        "django.contrib.contenttypes",
        "django.contrib.messages",
    ]
    EXTERNAL_APPS = [
        "rest_framework",  # pip install djangorestframework
    ]
    LOCAL_APPS = [
        "backend",
    ]
    INSTALLED_APPS = DEFAULT_APPS + EXTERNAL_APPS + LOCAL_APPS

    settings.configure(
        BASE_DIR=BASE_DIR,
        DEBUG=True,
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": os.path.join(BASE_DIR, "db.sqlite3"),
            }
        },
        INSTALLED_APPS=INSTALLED_APPS,
        # INSTALLED_APPS = (
        #     "backend",
        # ),
        TIME_ZONE="UTC",
        USE_TZ=True,
    )
    django.setup()


initialize_django()


# @timing
# def time_ruuid0():
#     [str(ruuid.uuid4()) for _ in range(10_000_000)]

# @timing
# def time_ruuid1():
#     [ruuid.simple() for _ in range(10_000_000)]

# @timing
# def time_uuid():
#     [str(uuid.uuid4()) for _ in range(10_000_000)]

# def main():
#     # _ruid0 = ruuid.uuid4()
#     # print(_ruid0, type(_ruid0))

#     # _ruid1 = ruuid.simple()
#     # print(_ruid1, type(_ruid1))

#     # _uid = uuid.uuid4()
#     # print(_uid, type(_uid))

#     time_ruuid0()
#     print()
#     time_ruuid1()
#     print()
#     time_uuid()


def main():
    pass
    # dp = backend.DataStructures.DataPoint(
    #     latitude = 10,
    #     longitude = 55,
    #     external_timestamp = datetime.now()
    # )
    # print("DataPoint: ", dp)
    # print("Hello World!")


if __name__ == "__main__":
    main()
