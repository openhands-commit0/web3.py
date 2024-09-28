import time
from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, Optional
from web3.exceptions import StaleBlockchain
from web3.types import AsyncMiddleware, AsyncMiddlewareCoroutine, BlockData, Middleware, RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3
SKIP_STALECHECK_FOR_METHODS = ('eth_getBlockByNumber',)

def make_stalecheck_middleware(allowable_delay: int, skip_stalecheck_for_methods: Collection[str]=SKIP_STALECHECK_FOR_METHODS) -> Middleware:
    """
    Use to require that a function will run only of the blockchain is recently updated.

    This middleware takes an argument, so unlike other middleware, you must make the
    middleware with a method call.

    For example: `make_stalecheck_middleware(60*5)`

    If the latest block in the chain is older than 5 minutes in this example, then the
    middleware will raise a StaleBlockchain exception.
    """
    pass

async def async_make_stalecheck_middleware(allowable_delay: int, skip_stalecheck_for_methods: Collection[str]=SKIP_STALECHECK_FOR_METHODS) -> AsyncMiddleware:
    """
    Use to require that a function will run only of the blockchain is recently updated.

    This middleware takes an argument, so unlike other middleware, you must make the
    middleware with a method call.

    For example: `async_make_stalecheck_middleware(60*5)`

    If the latest block in the chain is older than 5 minutes in this example, then the
    middleware will raise a StaleBlockchain exception.
    """
    pass