from typing import TYPE_CHECKING
from web3.types import FeeHistory, Wei
if TYPE_CHECKING:
    from web3.eth import AsyncEth
    from web3.eth import Eth
PRIORITY_FEE_MAX = Wei(1500000000)
PRIORITY_FEE_MIN = Wei(1000000000)
PRIORITY_FEE_HISTORY_PARAMS = (10, 'pending', [5.0])