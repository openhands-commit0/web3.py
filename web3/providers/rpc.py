import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from eth_typing import URI
from eth_utils import to_dict
from web3._utils.http import construct_user_agent
from web3._utils.request import cache_and_return_session, get_default_http_endpoint, make_post_request
from web3.datastructures import NamedElementOnion
from web3.middleware import http_retry_request_middleware
from web3.types import Middleware, RPCEndpoint, RPCResponse
from .base import JSONBaseProvider

class HTTPProvider(JSONBaseProvider):
    logger = logging.getLogger('web3.providers.HTTPProvider')
    endpoint_uri = None
    _request_args = None
    _request_kwargs = None
    _middlewares: Tuple[Middleware, ...] = NamedElementOnion([(http_retry_request_middleware, 'http_retry_request')])

    def __init__(self, endpoint_uri: Optional[Union[URI, str]]=None, request_kwargs: Optional[Any]=None, session: Optional[Any]=None) -> None:
        if endpoint_uri is None:
            self.endpoint_uri = get_default_http_endpoint()
        else:
            self.endpoint_uri = URI(endpoint_uri)
        self._request_kwargs = request_kwargs or {}
        if session:
            cache_and_return_session(self.endpoint_uri, session)
        super().__init__()

    def __str__(self) -> str:
        return f'RPC connection {self.endpoint_uri}'