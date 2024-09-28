import collections
import itertools
from typing import Any, Callable
from eth_typing import Hash32
from web3 import Web3
from web3.types import RPCEndpoint, RPCResponse, TxReceipt
counter = itertools.count()
INVOCATIONS_BEFORE_RESULT = 5