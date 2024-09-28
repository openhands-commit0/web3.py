from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence, Union
from toolz import merge
from web3._utils.normalizers import abi_ens_resolver, async_abi_ens_resolver
from web3._utils.rpc_abi import RPC_ABIS, abi_request_formatters
from web3.types import AsyncMiddlewareCoroutine, Middleware, RPCEndpoint
from .._utils.abi import abi_data_tree, async_data_tree_map, strip_abi_type
from .._utils.formatters import recursive_map
from .formatting import construct_formatting_middleware
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3