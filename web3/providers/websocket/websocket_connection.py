from typing import TYPE_CHECKING, Any, Dict
from web3.types import RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3.main import _PersistentConnectionWeb3
    from web3.manager import _AsyncPersistentMessageStream

class WebsocketConnection:
    """
    A class that houses the public API for interacting with the websocket connection
    via a `_PersistentConnectionWeb3` instance.
    """

    def __init__(self, w3: '_PersistentConnectionWeb3'):
        self._manager = w3.manager