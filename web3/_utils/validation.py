import itertools
from typing import Any, Dict
from eth_typing import HexStr, TypeStr
from eth_utils import function_abi_to_4byte_selector, is_0x_prefixed, is_binary_address, is_boolean, is_bytes, is_checksum_address, is_dict, is_hex_address, is_integer, is_list_like, is_string
from eth_utils.curried import apply_formatter_to_array
from eth_utils.hexadecimal import encode_hex
from eth_utils.toolz import compose, groupby, valfilter, valmap
from ens.utils import is_valid_ens_name
from web3._utils.abi import abi_to_signature, filter_by_type, is_address_type, is_array_type, is_bool_type, is_bytes_type, is_int_type, is_recognized_type, is_string_type, is_uint_type, length_of_array_type, sub_type_of_array_type
from web3.exceptions import InvalidAddress
from web3.types import ABI, ABIFunction

def validate_abi(abi: ABI) -> None:
    """
    Helper function for validating an ABI
    """
    pass

def validate_abi_type(abi_type: TypeStr) -> None:
    """
    Helper function for validating an abi_type
    """
    pass

def validate_abi_value(abi_type: TypeStr, value: Any) -> None:
    """
    Helper function for validating a value against the expected abi_type
    Note: abi_type 'bytes' must either be python3 'bytes' object or ''
    """
    pass

def validate_address(value: Any) -> None:
    """
    Helper function for validating an address
    """
    pass