import asyncio
import pytest
from typing import TYPE_CHECKING, Any, Dict, Tuple, cast
from eth_utils import is_hexstr
from hexbytes import HexBytes
from web3.datastructures import AttributeDict
from web3.middleware import async_geth_poa_middleware
from web3.types import FormattedEthSubscriptionResponse
if TYPE_CHECKING:
    from web3.main import _PersistentConnectionWeb3

class PersistentConnectionProviderTest:
    pass