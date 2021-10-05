# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
# ------------------------------------------------------------------------------#
#
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import json

from channels.db import database_sync_to_async
# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django extensions              #
# ------------------------------------------------------------------------------#
from channels.generic.websocket import AsyncJsonWebsocketConsumer

# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
# from asgiref.sync import async_to_sync
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
from .models import DataPoint
from .serializers import DataPointSerializer
from .utils import coords_to_geojson, handle_datapoints

# ------------------------------------------------------------------------------#
#                 Import third-party libraries: Django std. lib.               #
# ------------------------------------------------------------------------------#


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
class EventsConsumer(AsyncJsonWebsocketConsumer):
    """"""

    # --------------------------------------------------------------------------#
    # Main class methods                                                       #
    # --------------------------------------------------------------------------#
    async def connect(self):
        """
        Description:
            Default method that is called whenever a user establishes a
            websocket connection.
        Args:
            None
        Returns:
            None
        """
        # user = self.scope["user"]
        # if user.is_anonymous: # Disconnect anonymous users
        #     await self.close()
        # else:
        # Accept the connection
        await self.accept()
        print("The user connected successfully!")

    async def receive(self, text_data):
        """
        Description:
            Default method for receiving incoming messages from connected users.
            Whenever a new message is received, then convert it to json data and
            send a reponse.
        Args:
            None
        Returns:
            None
        """
        # Parse the received data
        data = json.loads(text_data)
        await self.respond(data)

    async def disconnect(self, close_code):
        """"""
        # Leave rooms on disconnect
        print("The user disconnected successfully...")

    # --------------------------------------------------------------------------#
    # Websocket response handling                                              #
    # --------------------------------------------------------------------------#
    async def respond(self, data):
        """Respond to the received message."""
        ####
        #### DataPoint methods
        ####
        if data["type"] == "retrieve-datapoints":
            return await self.retrieve_datapoints(data)
        elif data["type"] == "retrieve-segments":
            return await self.retrieve_segments(data)
        print("Do nothing...")

    # async def retrieve_datapoints(self, data):
    #     qs = await database_sync_to_async(
    #         DataPoint.objects.all
    #     )()
    #     serializer = await database_sync_to_async(
    #         DataPointSerializer
    #     )(instance = qs, many = True)
    #     return_data = await database_sync_to_async(getattr)(serializer, "data")
    #     return_data = await database_sync_to_async(coords_to_geojson)(return_data)
    #     message = {
    #         "type": "return-datapoints",
    #         "data": return_data,
    #     }
    #     print(message)
    #     await self.send_json(message)

    async def retrieve_datapoints(self, data):
        qs = await database_sync_to_async(DataPoint.objects.all)()
        serializer = await database_sync_to_async(DataPointSerializer)(
            instance=qs, many=True
        )
        return_data = await database_sync_to_async(getattr)(serializer, "data")
        return_data = await database_sync_to_async(coords_to_geojson)(
            return_data
        )
        message = {
            "type": "return-datapoints",
            "data": return_data,
        }
        print(message)
        await self.send_json(message)

    async def retrieve_segments(self, data):
        qs = await database_sync_to_async(DataPoint.objects.all)()
        # qs = qs[:250] # For testing...
        serializer = await database_sync_to_async(DataPointSerializer)(
            instance=qs, many=True
        )
        return_data = await database_sync_to_async(getattr)(serializer, "data")
        return_data0, return_data1 = await database_sync_to_async(
            handle_datapoints
        )(return_data)
        message = {
            "type": "return-segments",
            "data": return_data0,
            "extra_data": return_data1,
        }
        await self.send_json(message)
