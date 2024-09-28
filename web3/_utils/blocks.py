from typing import Any, Optional
from eth_utils import is_bytes, is_hex, is_integer, is_string, is_text, remove_0x_prefix
from eth_utils.toolz import curry
from web3.types import RPCEndpoint