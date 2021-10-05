from django.conf.urls import url

from . import consumers

websocket_urlpatterns = [
    url(r"^api/v1/ws/events", consumers.EventsConsumer),
]
