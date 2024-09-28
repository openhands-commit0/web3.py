import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, List, Optional, Sequence, Tuple, Union, cast
from eth_utils.toolz import pipe
from hexbytes import HexBytes
from websockets.exceptions import ConnectionClosedOK
from web3._utils.caching import generate_cache_key
from web3._utils.compat import Self
from web3.datastructures import NamedElementOnion
from web3.exceptions import BadResponseFormat, MethodUnavailable, ProviderConnectionError, TaskNotRunning
from web3.middleware import abi_middleware, async_attrdict_middleware, async_buffered_gas_estimate_middleware, async_gas_price_strategy_middleware, async_name_to_address_middleware, async_validation_middleware, attrdict_middleware, buffered_gas_estimate_middleware, gas_price_strategy_middleware, name_to_address_middleware, validation_middleware
from web3.module import apply_result_formatters
from web3.providers import AutoProvider, PersistentConnectionProvider
from web3.types import AsyncMiddleware, AsyncMiddlewareOnion, Middleware, MiddlewareOnion, RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3.main import AsyncWeb3, Web3
    from web3.providers import AsyncBaseProvider, BaseProvider
    from web3.providers.websocket.request_processor import RequestProcessor
NULL_RESPONSES = [None, HexBytes('0x'), '0x']
METHOD_NOT_FOUND = -32601

class RequestManager:
    logger = logging.getLogger('web3.RequestManager')
    middleware_onion: Union[MiddlewareOnion, AsyncMiddlewareOnion, NamedElementOnion[None, None]]

    def __init__(self, w3: Union['AsyncWeb3', 'Web3'], provider: Optional[Union['BaseProvider', 'AsyncBaseProvider']]=None, middlewares: Optional[Union[Sequence[Tuple[Middleware, str]], Sequence[Tuple[AsyncMiddleware, str]]]]=None) -> None:
        self.w3 = w3
        if provider is None:
            self.provider = AutoProvider()
        else:
            self.provider = provider
        if middlewares is None:
            middlewares = self.async_default_middlewares() if self.provider.is_async else self.default_middlewares(cast('Web3', w3))
        self.middleware_onion = NamedElementOnion(middlewares)
        if isinstance(provider, PersistentConnectionProvider):
            provider = cast(PersistentConnectionProvider, self.provider)
            self._request_processor: RequestProcessor = provider._request_processor
    w3: Union['AsyncWeb3', 'Web3'] = None
    _provider = None

    @staticmethod
    def default_middlewares(w3: 'Web3') -> List[Tuple[Middleware, str]]:
        """
        List the default middlewares for the request manager.
        Leaving w3 unspecified will prevent the middleware from resolving names.
        Documentation should remain in sync with these defaults.
        """
        pass

    @staticmethod
    def async_default_middlewares() -> List[Tuple[AsyncMiddleware, str]]:
        """
        List the default async middlewares for the request manager.
        Documentation should remain in sync with these defaults.
        """
        pass

    def request_blocking(self, method: Union[RPCEndpoint, Callable[..., RPCEndpoint]], params: Any, error_formatters: Optional[Callable[..., Any]]=None, null_result_formatters: Optional[Callable[..., Any]]=None) -> Any:
        """
        Make a synchronous request using the provider
        """
        pass

    async def coro_request(self, method: Union[RPCEndpoint, Callable[..., RPCEndpoint]], params: Any, error_formatters: Optional[Callable[..., Any]]=None, null_result_formatters: Optional[Callable[..., Any]]=None) -> Any:
        """
        Coroutine for making a request using the provider
        """
        pass

class _AsyncPersistentMessageStream:
    """
    Async generator for pulling subscription responses from the request processor
    subscription queue. This abstraction is necessary to define the `__aiter__()`
    method required for use with "async for" loops.
    """

    def __init__(self, manager: RequestManager, *args: Any, **kwargs: Any) -> None:
        self.manager = manager
        self.provider: PersistentConnectionProvider = cast(PersistentConnectionProvider, manager._provider)
        super().__init__(*args, **kwargs)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> RPCResponse:
        try:
            return await self.manager._get_next_ws_message()
        except ConnectionClosedOK:
            raise StopAsyncIteration