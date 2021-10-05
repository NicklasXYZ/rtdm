import backend.routing
from channels.routing import ProtocolTypeRouter, URLRouter

application = ProtocolTypeRouter(
    {"websocket": URLRouter(backend.routing.websocket_urlpatterns,),}
)
