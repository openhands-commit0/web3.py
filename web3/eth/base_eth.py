from typing import Any, List, NoReturn, Optional, Tuple, Union
from eth_account import Account
from eth_typing import Address, ChecksumAddress, HexStr
from eth_utils import is_checksum_address, is_string
from eth_utils.toolz import assoc
from web3._utils.empty import Empty, empty
from web3._utils.encoding import to_hex
from web3.module import Module
from web3.types import ENS, BlockIdentifier, CallOverride, FilterParams, GasPriceStrategy, TxParams, Wei

class BaseEth(Module):
    _default_account: Union[ChecksumAddress, Empty] = empty
    _default_block: BlockIdentifier = 'latest'
    _default_contract_factory: Any = None
    _gas_price_strategy = None
    is_async = False
    account = Account()