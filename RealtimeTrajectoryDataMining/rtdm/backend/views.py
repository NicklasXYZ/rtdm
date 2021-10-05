from django.apps import apps
from django.conf import settings
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .serializers import CustomBaseUserSerializer


class UserView(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    mixins.DestroyModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    model = apps.get_model(settings.APP_NAME, "CustomBaseUser")
    queryset = model.objects.all()
    serializer_class = CustomBaseUserSerializer
    pk_url_kwarg = "pk"
    lookup_field = "pk"

    def get_instance(self):
        return self.request.user

    def get_permissions(self):
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
