from abc import ABC
import asyncio
import logging
from typing import Optional
from websockets import ConnectionClosed, ConnectionClosedOK, WebSocketClientProtocol, WebSocketException
from web3._utils.caching import generate_cache_key
from web3.exceptions import ProviderConnectionError, TaskNotRunning, TimeExhausted
from web3.providers.async_base import AsyncJSONBaseProvider
from web3.providers.websocket.request_processor import RequestProcessor
from web3.types import RPCId, RPCResponse
DEFAULT_PERSISTENT_CONNECTION_TIMEOUT = 50.0

class PersistentConnectionProvider(AsyncJSONBaseProvider, ABC):
    logger = logging.getLogger('web3.providers.PersistentConnectionProvider')
    has_persistent_connection = True
    endpoint_uri: Optional[str] = None
    _max_connection_retries: int = 5
    _ws: Optional[WebSocketClientProtocol] = None
    _request_processor: RequestProcessor
    _message_listener_task: Optional['asyncio.Task[None]'] = None
    _listen_event: asyncio.Event = asyncio.Event()

    def __init__(self, request_timeout: float=DEFAULT_PERSISTENT_CONNECTION_TIMEOUT, subscription_response_queue_size: int=500, request_information_cache_size: int=500, silence_listener_task_exceptions: bool=False) -> None:
        super().__init__()
        self._request_processor = RequestProcessor(self, subscription_response_queue_size=subscription_response_queue_size, request_information_cache_size=request_information_cache_size)
        self.request_timeout = request_timeout
        self.silence_listener_task_exceptions = silence_listener_task_exceptions

    def _error_log_listener_task_exception(self, e: Exception) -> None:
        """
        When silencing listener task exceptions, this method is used to log the
        exception and keep the listener task alive. Override this method to fine-tune
        error logging behavior for the implementation class.
        """
        pass

    def _handle_listener_task_exceptions(self) -> None:
        """
        Should be called every time a `PersistentConnectionProvider` is polling for
        messages in the main loop. If the message listener task has completed and an
        exception was recorded, raise the exception in the main loop.
        """
        pass