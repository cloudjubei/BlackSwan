import asyncio
import time
import socketio
import tornado
import tornado.ioloop
import tornado.web
from config import EMITTER_PORT

class WebsocketServer:

    def __init__(self, identifier: str, interval: str, port: int = EMITTER_PORT):
        self.port = port
        self.sio = socketio.AsyncServer(async_mode='tornado')
        self.identifier = identifier
        self.interval = interval

        self.latest_action = 0
        self.latest_signal = {
            "identifier": "_",
            "tokenPair": "_",
            "interval": "_",
            "timestamp": 0,
            "action": 0,
            "certainty": 0
        }
    
    async def start(self):
        self.__setup_callbacks()

        app = tornado.web.Application(
            [
                (r"/socket.io/", socketio.get_tornado_handler(self.sio)),
            ]
        )
        app.listen(self.port)
        print(f'WS Server started at http://localhost:{self.port}')

        await asyncio.Event().wait()  # Keep the server running

    async def emit_signal(self, tokenPair, action):
        self.latest_action = action
        self.latest_signal = {
            "identifier": self.identifier,
            "tokenPair": tokenPair,
            "interval": self.interval,
            "timestamp": time.time() * 1000,
            "action": action,
            "certainty": 1
        }
        print(f"WS Server emitting: {self.latest_signal}")
        await self.sio.emit(f'{self.identifier}-{tokenPair}-{self.interval}', self.latest_signal)

    def __setup_callbacks(self):
        @self.sio.event
        async def connect(sid, env):
            print(f"Client connected: {sid}")
        @self.sio.event
        async def disconnect(sid):
            print(f"Client disconnected: {sid}")

        @self.sio.event
        def identity_type(sid, data):
            return 'signal'

        @self.sio.event
        def identity_identifier(sid, data):
            return 'ai-model'

        @self.sio.event
        def identity_tokens(sid, data):
            return ["BTCUSDT"]

        @self.sio.event
        def signal_latest(sid, tokenPair):
            print(f'got asked signal_latest sid: {sid} tokenPair: ', tokenPair)
            return {
                "identifier": self.identifier,
                "tokenPair": tokenPair,
                "interval": self.interval,
                "timestamp": time.time() * 1000,
                "action": self.latest_action,
                "certainty": 1
            }
            return self.latest_signal
