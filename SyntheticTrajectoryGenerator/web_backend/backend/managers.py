# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django std. lib.               #
# ------------------------------------------------------------------------------#
from django.contrib.auth.base_user import BaseUserManager
from django.db import models

# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django extensions              #
# ------------------------------------------------------------------------------#
#                                                                              #


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class DataPointManager(models.Manager):
    pass


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class TrackManager(models.Manager):
    pass


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class CustomBaseUserManager(BaseUserManager):
    """"""

    # --------------------------------------------------------------------------#
    # Methods                                                                  #
    # --------------------------------------------------------------------------#
    def create_user(self, user_data):
        """
        Description:
            Create and save a new user.

        Args:
            **user_data (dict):

        Returns:
            user ():
        """
        user_data["email"] = self.normalize_email(user_data["email"])
        password = user_data.pop("password")
        user = self.model(**user_data)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email, password, **extra_fields):
        """
        Description:
            Create and save a new superuser.

        Args:
            email           (str):
            password        (str):
            **extra_fields (dict):

        Returns:
            user ():
        """
        # These fields should be set to the following values
        # regardless of what is given as input
        extra_fields["is_admin"] = True
        extra_fields["is_superuser"] = True
        extra_fields["is_staff"] = True
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save()
        return user
