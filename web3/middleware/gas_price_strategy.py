from typing import TYPE_CHECKING, Any, Callable
from eth_utils.toolz import assoc
from web3._utils.method_formatters import to_hex_if_integer
from web3._utils.utility_methods import all_in_dict, any_in_dict, none_in_dict
from web3.constants import DYNAMIC_FEE_TXN_PARAMS
from web3.exceptions import InvalidTransaction, TransactionTypeMismatch
from web3.types import AsyncMiddlewareCoroutine, BlockData, RPCEndpoint, RPCResponse, TxParams, Wei
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3

def gas_price_strategy_middleware(make_request: Callable[[RPCEndpoint, Any], Any], w3: 'Web3') -> Callable[[RPCEndpoint, Any], RPCResponse]:
    """
    - Uses a gas price strategy if one is set. This is only supported
      for legacy transactions. It is recommended to send dynamic fee
      transactions (EIP-1559) whenever possible.

    - Validates transaction params against legacy and dynamic fee txn values.
    """
    pass

async def async_gas_price_strategy_middleware(make_request: Callable[[RPCEndpoint, Any], Any], async_w3: 'AsyncWeb3') -> AsyncMiddlewareCoroutine:
    """
    - Uses a gas price strategy if one is set. This is only supported for
      legacy transactions. It is recommended to send dynamic fee transactions
      (EIP-1559) whenever possible.

    - Validates transaction params against legacy and dynamic fee txn values.
    """
    pass