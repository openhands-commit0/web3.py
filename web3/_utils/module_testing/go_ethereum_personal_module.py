import json
import pytest
from typing import TYPE_CHECKING, cast
from eth_typing import ChecksumAddress
from eth_utils import is_checksum_address, is_list_like, is_same_address, is_string
from hexbytes import HexBytes
from web3 import constants
from web3.datastructures import AttributeDict
from web3.types import TxParams, Wei
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3
PRIVATE_KEY_HEX = '0x56ebb41875ceedd42e395f730e03b5c44989393c9f0484ee6bc05f933673458f'
SECOND_PRIVATE_KEY_HEX = '0x56ebb41875ceedd42e395f730e03b5c44989393c9f0484ee6bc05f9336712345'
THIRD_PRIVATE_KEY_HEX = '0x56ebb41875ceedd42e395f730e03b5c44989393c9f0484ee6bc05f9336754321'
PASSWORD = 'web3-testing'
ADDRESS = '0x844B417c0C58B02c2224306047B9fb0D3264fE8c'
SECOND_ADDRESS = '0xB96b6B21053e67BA59907E252D990C71742c41B8'
PRIVATE_KEY_FOR_UNLOCK = '0x392f63a79b1ff8774845f3fa69de4a13800a59e7083f5187f1558f0797ad0f01'
ACCOUNT_FOR_UNLOCK = '0x12efDc31B1a8FA1A1e756DFD8A1601055C971E13'

class GoEthereumPersonalModuleTest:
    pass

class GoEthereumAsyncPersonalModuleTest:
    pass