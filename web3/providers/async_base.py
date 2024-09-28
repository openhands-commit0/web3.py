import itertools
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Sequence, Tuple, cast
from eth_utils import is_text, to_bytes, to_text
from web3._utils.encoding import FriendlyJsonSerde, Web3JsonEncoder
from web3.exceptions import ProviderConnectionError
from web3.middleware import async_combine_middlewares
from web3.types import AsyncMiddleware, AsyncMiddlewareOnion, MiddlewareOnion, RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3 import AsyncWeb3, WebsocketProviderV2

class AsyncBaseProvider:
    _middlewares: Tuple[AsyncMiddleware, ...] = ()
    _request_func_cache: Tuple[Tuple[AsyncMiddleware, ...], Callable[..., Coroutine[Any, Any, RPCResponse]]] = (None, None)
    is_async = True
    has_persistent_connection = False
    global_ccip_read_enabled: bool = True
    ccip_read_max_redirects: int = 4

class AsyncJSONBaseProvider(AsyncBaseProvider):

    def __init__(self) -> None:
        super().__init__()
        self.request_counter = itertools.count()