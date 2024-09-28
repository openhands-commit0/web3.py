from typing import TYPE_CHECKING, Optional, cast
from eth_typing import ChecksumAddress
from eth_utils.toolz import assoc, merge
from hexbytes import HexBytes
from web3._utils.transactions import prepare_replacement_transaction
from web3._utils.utility_methods import any_in_dict
from web3.constants import DYNAMIC_FEE_TXN_PARAMS
from web3.types import BlockIdentifier, TxData, TxParams, Wei, _Hash32
if TYPE_CHECKING:
    from web3.eth import AsyncEth
    from web3.main import AsyncWeb3
TRANSACTION_DEFAULTS = {'value': 0, 'data': b'', 'gas': _estimate_gas, 'gasPrice': lambda async_w3, tx: async_w3.eth.generate_gas_price(tx), 'maxFeePerGas': _max_fee_per_gas, 'maxPriorityFeePerGas': _max_priority_fee_gas, 'chainId': _chain_id}

async def async_fill_transaction_defaults(async_w3: 'AsyncWeb3', transaction: TxParams) -> TxParams:
    """
    if async_w3 is None, fill as much as possible while offline
    """
    pass