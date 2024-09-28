from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type
from eth_utils.toolz import excepts
from web3.types import Middleware, RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3 import Web3