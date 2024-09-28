import asyncio
import json
import logging
import os
from threading import Thread
from types import TracebackType
from typing import Any, Optional, Type, Union
from eth_typing import URI
from websockets.client import connect
from websockets.legacy.client import WebSocketClientProtocol
from web3.exceptions import Web3ValidationError
from web3.providers.base import JSONBaseProvider
from web3.types import RPCEndpoint, RPCResponse
RESTRICTED_WEBSOCKET_KWARGS = {'uri', 'loop'}
DEFAULT_WEBSOCKET_TIMEOUT = 10

class PersistentWebSocket:

    def __init__(self, endpoint_uri: URI, websocket_kwargs: Any) -> None:
        self.ws: Optional[WebSocketClientProtocol] = None
        self.endpoint_uri = endpoint_uri
        self.websocket_kwargs = websocket_kwargs

    async def __aenter__(self) -> WebSocketClientProtocol:
        if self.ws is None:
            self.ws = await connect(uri=self.endpoint_uri, **self.websocket_kwargs)
        return self.ws

    async def __aexit__(self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        if exc_val is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

class WebsocketProvider(JSONBaseProvider):
    logger = logging.getLogger('web3.providers.WebsocketProvider')
    _loop = None

    def __init__(self, endpoint_uri: Optional[Union[URI, str]]=None, websocket_kwargs: Optional[Any]=None, websocket_timeout: int=DEFAULT_WEBSOCKET_TIMEOUT) -> None:
        self.endpoint_uri = URI(endpoint_uri)
        self.websocket_timeout = websocket_timeout
        if self.endpoint_uri is None:
            self.endpoint_uri = get_default_endpoint()
        if WebsocketProvider._loop is None:
            WebsocketProvider._loop = _get_threaded_loop()
        if websocket_kwargs is None:
            websocket_kwargs = {}
        else:
            found_restricted_keys = set(websocket_kwargs).intersection(RESTRICTED_WEBSOCKET_KWARGS)
            if found_restricted_keys:
                raise Web3ValidationError(f'{RESTRICTED_WEBSOCKET_KWARGS} are not allowed in websocket_kwargs, found: {found_restricted_keys}')
        self.conn = PersistentWebSocket(self.endpoint_uri, websocket_kwargs)
        super().__init__()

    def __str__(self) -> str:
        return f'WS connection {self.endpoint_uri}'