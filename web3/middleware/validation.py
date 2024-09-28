from typing import TYPE_CHECKING, Any, Callable, Dict
from eth_utils.curried import apply_formatter_at_index, apply_formatter_if, apply_formatters_to_dict, is_null, is_string
from eth_utils.toolz import complement, compose, curry, dissoc
from hexbytes import HexBytes
from web3._utils.formatters import hex_to_integer
from web3._utils.rpc_abi import RPC
from web3.exceptions import ExtraDataLengthError, Web3ValidationError
from web3.middleware.formatting import async_construct_web3_formatting_middleware, construct_web3_formatting_middleware
from web3.types import AsyncMiddlewareCoroutine, Formatters, FormattersDict, RPCEndpoint, TxParams
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3
MAX_EXTRADATA_LENGTH = 32
is_not_null = complement(is_null)
to_integer_if_hex = apply_formatter_if(is_string, hex_to_integer)
BLOCK_VALIDATORS = {'extraData': _check_extradata_length}
block_validator = apply_formatter_if(is_not_null, apply_formatters_to_dict(BLOCK_VALIDATORS))
METHODS_TO_VALIDATE = [RPC.eth_sendTransaction, RPC.eth_estimateGas, RPC.eth_call, RPC.eth_createAccessList]
validation_middleware = construct_web3_formatting_middleware(build_method_validators)