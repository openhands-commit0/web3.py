import functools
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union, cast
from eth_abi.codec import ABICodec
from eth_abi.registry import registry as default_registry
from eth_typing import ChecksumAddress, HexStr, TypeStr
from eth_utils import add_0x_prefix, encode_hex, function_abi_to_4byte_selector, is_binary_address, is_checksum_address, is_list_like, is_text
from eth_utils.toolz import pipe
from hexbytes import HexBytes
from web3._utils.abi import abi_to_signature, check_if_arguments_can_be_encoded, filter_by_argument_count, filter_by_argument_name, filter_by_encodability, filter_by_name, filter_by_type, get_abi_input_types, get_aligned_abi_inputs, get_fallback_func_abi, get_receive_func_abi, map_abi_data, merge_args_and_kwargs, named_tree
from web3._utils.blocks import is_hex_encoded_block_hash
from web3._utils.encoding import to_hex
from web3._utils.function_identifiers import FallbackFn, ReceiveFn
from web3._utils.method_formatters import to_integer_if_hex
from web3._utils.normalizers import abi_address_to_hex, abi_bytes_to_bytes, abi_ens_resolver, abi_string_to_text
from web3.exceptions import BlockNumberOutofRange, Web3ValidationError
from web3.types import ABI, ABIEvent, ABIFunction, BlockIdentifier, BlockNumber, TxParams
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3

def extract_argument_types(*args: Sequence[Any]) -> str:
    """
    Takes a list of arguments and returns a string representation of the argument types,
    appropriately collapsing `tuple` types into the respective nested types.
    """
    pass

def prepare_transaction(address: ChecksumAddress, w3: Union['AsyncWeb3', 'Web3'], fn_identifier: Union[str, Type[FallbackFn], Type[ReceiveFn]], contract_abi: Optional[ABI]=None, fn_abi: Optional[ABIFunction]=None, transaction: Optional[TxParams]=None, fn_args: Optional[Sequence[Any]]=None, fn_kwargs: Optional[Any]=None) -> TxParams:
    """
    :parameter `is_function_abi` is used to distinguish  function abi from contract abi
    Returns a dictionary of the transaction that could be used to call this
    TODO: make this a public API
    TODO: add new prepare_deploy_transaction API
    """
    pass

def validate_payable(transaction: TxParams, abi: ABIFunction) -> None:
    """Raise Web3ValidationError if non-zero ether
    is sent to a non-payable function.
    """
    pass