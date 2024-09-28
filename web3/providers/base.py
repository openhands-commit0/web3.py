import itertools
from typing import TYPE_CHECKING, Any, Callable, Sequence, Tuple, cast
from eth_utils import to_bytes, to_text
from web3._utils.encoding import FriendlyJsonSerde, Web3JsonEncoder
from web3.exceptions import ProviderConnectionError
from web3.middleware import combine_middlewares
from web3.types import Middleware, MiddlewareOnion, RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3 import Web3

class BaseProvider:
    _middlewares: Tuple[Middleware, ...] = ()
    _request_func_cache: Tuple[Tuple[Middleware, ...], Callable[..., RPCResponse]] = (None, None)
    is_async = False
    has_persistent_connection = False
    global_ccip_read_enabled: bool = True
    ccip_read_max_redirects: int = 4

    def request_func(self, w3: 'Web3', outer_middlewares: MiddlewareOnion) -> Callable[..., RPCResponse]:
        """
        @param outer_middlewares is an iterable of middlewares,
            ordered by first to execute
        @returns a function that calls all the middleware and
            eventually self.make_request()
        """
        pass

class JSONBaseProvider(BaseProvider):

    def __init__(self) -> None:
        self.request_counter = itertools.count()