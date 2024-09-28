from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional, cast
from eth_utils.toolz import assoc, curry, merge
from web3.types import AsyncMiddleware, AsyncMiddlewareCoroutine, EthSubscriptionParams, Formatters, FormattersDict, Literal, Middleware, RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3
    from web3.providers import PersistentConnectionProvider
FORMATTER_DEFAULTS: FormattersDict = {'request_formatters': {}, 'result_formatters': {}, 'error_formatters': {}}