from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, cast
from web3.types import AsyncMiddleware, AsyncMiddlewareCoroutine, Middleware, RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3.main import AsyncWeb3, Web3
    from web3.providers import PersistentConnectionProvider

def construct_fixture_middleware(fixtures: Dict[RPCEndpoint, Any]) -> Middleware:
    """
    Constructs a middleware which returns a static response for any method
    which is found in the provided fixtures.
    """
    pass

def construct_result_generator_middleware(result_generators: Dict[RPCEndpoint, Any]) -> Middleware:
    """
    Constructs a middleware which intercepts requests for any method found in
    the provided mapping of endpoints to generator functions, returning
    whatever response the generator function returns.  Callbacks must be
    functions with the signature `fn(method, params)`.
    """
    pass

def construct_error_generator_middleware(error_generators: Dict[RPCEndpoint, Any]) -> Middleware:
    """
    Constructs a middleware which intercepts requests for any method found in
    the provided mapping of endpoints to generator functions, returning
    whatever error message the generator function returns.  Callbacks must be
    functions with the signature `fn(method, params)`.
    """
    pass

async def async_construct_result_generator_middleware(result_generators: Dict[RPCEndpoint, Any]) -> AsyncMiddleware:
    """
    Constructs a middleware which returns a static response for any method
    which is found in the provided fixtures.
    """
    pass

async def async_construct_error_generator_middleware(error_generators: Dict[RPCEndpoint, Any]) -> AsyncMiddleware:
    """
    Constructs a middleware which intercepts requests for any method found in
    the provided mapping of endpoints to generator functions, returning
    whatever error message the generator function returns.  Callbacks must be
    functions with the signature `fn(method, params)`.
    """
    pass