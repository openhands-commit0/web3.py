import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional, Union
from eth_typing import URI
from toolz import merge
from websockets.client import connect
from websockets.exceptions import WebSocketException
from web3.exceptions import ProviderConnectionError, Web3ValidationError
from web3.providers.persistent import PersistentConnectionProvider
from web3.types import RPCEndpoint, RPCResponse
DEFAULT_PING_INTERVAL = 30
DEFAULT_PING_TIMEOUT = 300
VALID_WEBSOCKET_URI_PREFIXES = {'ws://', 'wss://'}
RESTRICTED_WEBSOCKET_KWARGS = {'uri', 'loop'}
DEFAULT_WEBSOCKET_KWARGS = {'ping_interval': DEFAULT_PING_INTERVAL, 'ping_timeout': DEFAULT_PING_TIMEOUT}

class WebsocketProviderV2(PersistentConnectionProvider):
    logger = logging.getLogger('web3.providers.WebsocketProviderV2')
    is_async: bool = True

    def __init__(self, endpoint_uri: Optional[Union[URI, str]]=None, websocket_kwargs: Optional[Dict[str, Any]]=None, silence_listener_task_exceptions: bool=False, **kwargs: Any) -> None:
        self.endpoint_uri = URI(endpoint_uri) if endpoint_uri is not None else get_default_endpoint()
        if not any((self.endpoint_uri.startswith(prefix) for prefix in VALID_WEBSOCKET_URI_PREFIXES)):
            raise Web3ValidationError(f"WebSocket endpoint uri must begin with 'ws://' or 'wss://': {self.endpoint_uri}")
        if websocket_kwargs is not None:
            found_restricted_keys = set(websocket_kwargs).intersection(RESTRICTED_WEBSOCKET_KWARGS)
            if found_restricted_keys:
                raise Web3ValidationError(f'Found restricted keys for websocket_kwargs: {found_restricted_keys}.')
        self.websocket_kwargs = merge(DEFAULT_WEBSOCKET_KWARGS, websocket_kwargs or {})
        super().__init__(silence_listener_task_exceptions=silence_listener_task_exceptions, **kwargs)

    def __str__(self) -> str:
        return f'WebSocket connection: {self.endpoint_uri}'