import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from aiohttp import ClientSession
from eth_typing import URI
from eth_utils import to_dict
from web3._utils.http import construct_user_agent
from web3._utils.request import async_cache_and_return_session as _async_cache_and_return_session, async_make_post_request, get_default_http_endpoint
from web3.types import AsyncMiddleware, RPCEndpoint, RPCResponse
from ..datastructures import NamedElementOnion
from ..middleware.exception_retry_request import async_http_retry_request_middleware
from .async_base import AsyncJSONBaseProvider

class AsyncHTTPProvider(AsyncJSONBaseProvider):
    logger = logging.getLogger('web3.providers.AsyncHTTPProvider')
    endpoint_uri = None
    _request_kwargs = None
    _middlewares: Tuple[AsyncMiddleware, ...] = NamedElementOnion([(async_http_retry_request_middleware, 'http_retry_request')])

    def __init__(self, endpoint_uri: Optional[Union[URI, str]]=None, request_kwargs: Optional[Any]=None) -> None:
        if endpoint_uri is None:
            self.endpoint_uri = get_default_http_endpoint()
        else:
            self.endpoint_uri = URI(endpoint_uri)
        self._request_kwargs = request_kwargs or {}
        super().__init__()

    def __str__(self) -> str:
        return f'RPC connection {self.endpoint_uri}'