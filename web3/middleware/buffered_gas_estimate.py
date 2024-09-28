from typing import TYPE_CHECKING, Any, Callable
from eth_utils.toolz import assoc
from web3._utils.async_transactions import get_buffered_gas_estimate as async_get_buffered_gas_estimate
from web3._utils.transactions import get_buffered_gas_estimate
from web3.types import AsyncMiddlewareCoroutine, RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3.main import AsyncWeb3, Web3