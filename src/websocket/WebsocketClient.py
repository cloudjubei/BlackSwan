import asyncio
import socketio
import json
from config import PRICE_PORT

class WebsocketClient:

    def __init__(self, host: str=f'http://localhost:{PRICE_PORT}'):
        self.host = host
        self.sio_client = socketio.AsyncClient()
        self.is_connected = False
    
    async def start(self):
        self.__setup_callbacks()

        print(f'WS Client connecting to {self.host}')
        await self.sio_client.connect(self.host)
        await self.sio_client.wait()

    def listen_to_price(self, tokenPair: str, callback):
        print(f"WS Client starting to listen to price on: {tokenPair}")
        self.sio_client.on(tokenPair, callback)

    async def ask_price(self, tokenPair: str, interval: str, callback):
        print('WS Client ask_price')
        data = json.dumps({
            "tokenPair": tokenPair,
            "interval": interval
        })
        await self.sio_client.emit("price_latestKline", data, callback= callback)
        print('WS Client ask_price done')

    def __setup_callbacks(self):
        @self.sio_client.event
        async def connect():
            print(f'WS Client connected to {self.host}')
            self.is_connected = True

        @self.sio_client.event
        async def disconnect():
            print(f'WS Client disconnected from {self.host}')
            self.is_connected = False

        @self.sio_client.event
        async def response(data):
            print(f"WS Client received response: {data}")



