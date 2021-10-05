# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
# ------------------------------------------------------------------------------#
#
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import os

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
#
# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django extensions              #
# ------------------------------------------------------------------------------#
#


# ------------------------------------------------------------------------------#
DEBUG = True
APP_NAME = "backend"
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join(BASE_DIR, "db.dev.sqlite3"),
        "OPTIONS": {"timeout": 1000,},
    },
}

DJANGO_BACKINGSTORE = os.environ.get("DJANGO_BACKINGSTORE", "")
if DJANGO_BACKINGSTORE == "redis":
    REDIS_NAME = os.environ.get("REDIS_NAME", "redis")
    REDIS_BIND = os.environ.get("REDIS_BIND", "localhost")
    REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
    CHANNEL_LAYERS = {
        "default": {
            "BACKEND": "channels_redis.core.RedisChannelLayer",
            "CONFIG": {"hosts": [(REDIS_NAME, REDIS_PORT)],},
        },
    }
else:
    CHANNEL_LAYERS = {
        "default": {"BACKEND": "channels.layers.InMemoryChannelLayer",},
    }

SECRET_KEY = "_"

AUTH_USER_MODEL = "backend.CustomBaseUser"

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# Application definition
DEFAULT_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

EXTERNAL_APPS = [
    "rest_framework",  # pip install djangorestframework
    "channels",  # pip install channels channels_redis
]

LOCAL_APPS = [
    "backend",
]

INSTALLED_APPS = DEFAULT_APPS + EXTERNAL_APPS + LOCAL_APPS

STATIC_URL = "/static/"

STATIC_ROOT = os.path.join(BASE_DIR, "static/")
STATICFILES_DIRS = (os.path.join(BASE_DIR, "static_dirs/"),)

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        # "rest_framework.permissions.IsAuthenticated", # Use this one in production!
        "rest_framework.permissions.AllowAny",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        # "rest_framework_simplejwt.authentication.JWTAuthentication", # Use this one in production!
        # "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",  # Disable browserble api
    ],
}

ROOT_URLCONF = "tsanomdet.urls"
WSGI_APPLICATION = "tsanomdet.wsgi.application"
ASGI_APPLICATION = "tsanomdet.routing.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",  # To keep the Browsable API
]

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",},
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

ALLOWED_HOSTS = ["*"]
