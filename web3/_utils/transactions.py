import math
from typing import TYPE_CHECKING, List, Optional, Union, cast
from eth_typing import ChecksumAddress
from eth_utils.toolz import assoc, curry, merge
from hexbytes import HexBytes
from web3._utils.compat import Literal
from web3._utils.utility_methods import all_in_dict, any_in_dict
from web3.constants import DYNAMIC_FEE_TXN_PARAMS
from web3.types import BlockIdentifier, TxData, TxParams, _Hash32
TX_PARAM_LITERALS = Literal['type', 'from', 'to', 'gas', 'maxFeePerGas', 'maxPriorityFeePerGas', 'gasPrice', 'value', 'data', 'nonce', 'chainId', 'accessList', 'maxFeePerBlobGas', 'blobVersionedHashes']
VALID_TRANSACTION_PARAMS: List[TX_PARAM_LITERALS] = ['type', 'from', 'to', 'gas', 'accessList', 'maxFeePerGas', 'maxPriorityFeePerGas', 'gasPrice', 'value', 'data', 'nonce', 'chainId', 'maxFeePerBlobGas', 'blobVersionedHashes']
TRANSACTION_DEFAULTS = {'value': 0, 'data': b'', 'gas': lambda w3, tx: w3.eth.estimate_gas(tx), 'gasPrice': lambda w3, tx: w3.eth.generate_gas_price(tx), 'maxFeePerGas': lambda w3, tx: w3.eth.max_priority_fee + 2 * w3.eth.get_block('latest')['baseFeePerGas'], 'maxPriorityFeePerGas': lambda w3, tx: w3.eth.max_priority_fee, 'chainId': lambda w3, tx: w3.eth.chain_id}
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3

@curry
def fill_transaction_defaults(w3: 'Web3', transaction: TxParams) -> TxParams:
    """
    if w3 is None, fill as much as possible while offline
    """
    pass