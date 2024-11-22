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

def validate_abi_item(abi_item: Dict[str, Any]) -> None:
    """
    Helper function for validating an ABI item
    """
    if 'type' not in abi_item:
        raise ValueError("'type' is required in the ABI item")

    abi_type = abi_item['type']
    if abi_type not in ('function', 'constructor', 'event', 'fallback', 'receive'):
        raise ValueError(f"'type' must be one of 'function', 'constructor', 'event', 'fallback', or 'receive'. Got {abi_type}")

    if abi_type != 'fallback' and abi_type != 'receive':
        if 'inputs' not in abi_item:
            raise ValueError("'inputs' is required in the ABI item")
        if not is_list_like(abi_item['inputs']):
            raise ValueError("'inputs' must be a list")

        for input_item in abi_item['inputs']:
            validate_abi_input_output(input_item)

        if abi_type in ('function', 'constructor'):
            if 'outputs' not in abi_item:
                raise ValueError("'outputs' is required in the ABI item")
            if not is_list_like(abi_item['outputs']):
                raise ValueError("'outputs' must be a list")

            for output_item in abi_item['outputs']:
                validate_abi_input_output(output_item)

def validate_abi_input_output(item: Dict[str, Any]) -> None:
    """
    Helper function for validating an ABI input or output item
    """
    if 'type' not in item:
        raise ValueError("'type' is required in the ABI input/output item")
    validate_abi_type(item['type'])

def validate_abi(abi: ABI) -> None:
    """
    Helper function for validating an ABI
    """
    if not is_list_like(abi):
        raise ValueError("'abi' is not a list")

    for abi_item in abi:
        if not is_dict(abi_item):
            raise ValueError("The elements of 'abi' are not all dictionaries")

        validate_abi_item(abi_item)

def validate_abi_type(abi_type: TypeStr) -> None:
    """
    Helper function for validating an abi_type
    """
    if not is_recognized_type(abi_type):
        raise ValueError(f"'{abi_type}' is not a recognized type")

    if is_array_type(abi_type):
        validate_abi_type(sub_type_of_array_type(abi_type))

def validate_abi_value(abi_type: TypeStr, value: Any) -> None:
    """
    Helper function for validating a value against the expected abi_type
    Note: abi_type 'bytes' must either be python3 'bytes' object or ''
    """
    if is_array_type(abi_type) and not is_list_like(value):
        raise TypeError(f"Value must be list-like for array type: {abi_type}")

    if is_array_type(abi_type):
        sub_type = sub_type_of_array_type(abi_type)
        for v in value:
            validate_abi_value(sub_type, v)
        return

    if is_bool_type(abi_type) and not is_boolean(value):
        raise TypeError(f"Value must be boolean for type: {abi_type}")

    elif is_uint_type(abi_type) or is_int_type(abi_type):
        if not is_integer(value):
            raise TypeError(f"Value must be integer for type: {abi_type}")

    elif is_address_type(abi_type):
        validate_address(value)

    elif is_bytes_type(abi_type):
        if not is_bytes(value) and not is_string(value):
            raise TypeError(f"Value must be bytes or string for type: {abi_type}")

    elif is_string_type(abi_type) and not is_string(value):
        raise TypeError(f"Value must be string for type: {abi_type}")

def validate_address(value: Any) -> None:
    """
    Helper function for validating an address
    """
    if not is_address(value):
        raise InvalidAddress("Value must be a valid address, zero-padded to 20 bytes")

def is_address(value: Any) -> bool:
    """
    Helper function for checking if a value is a valid address
    """
    if not is_string(value):
        return False

    if is_binary_address(value):
        return True

    if is_hex_address(value):
        return True

    if is_checksum_address(value):
        return True

    if is_valid_ens_name(value):
        return True

    return False