import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union
from urllib.parse import urlparse
from eth_typing import URI
from web3.exceptions import CannotHandleRequest
from web3.providers import BaseProvider, HTTPProvider, IPCProvider, WebsocketProvider
from web3.types import RPCEndpoint, RPCResponse
HTTP_SCHEMES = {'http', 'https'}
WS_SCHEMES = {'ws', 'wss'}

class AutoProvider(BaseProvider):
    default_providers = (load_provider_from_environment, IPCProvider, HTTPProvider, WebsocketProvider)
    _active_provider = None

    def __init__(self, potential_providers: Optional[Sequence[Union[Callable[..., BaseProvider], Type[BaseProvider]]]]=None) -> None:
        """
        :param iterable potential_providers: ordered series of provider classes
            to attempt with

        AutoProvider will initialize each potential provider (without arguments),
        in an attempt to find an active node. The list will default to
        :attribute:`default_providers`.
        """
        if potential_providers:
            self._potential_providers = potential_providers
        else:
            self._potential_providers = self.default_providers