import json
import re
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Type, Union
from eth_abi.encoding import BaseArrayEncoder
from eth_typing import HexStr, Primitives, TypeStr
from eth_utils import add_0x_prefix, encode_hex, is_bytes, is_hex, is_list_like, remove_0x_prefix, to_bytes, to_hex
from eth_utils.toolz import curry
from hexbytes import HexBytes
from web3._utils.abi import is_address_type, is_array_type, is_bool_type, is_bytes_type, is_int_type, is_string_type, is_uint_type, size_of_type, sub_type_of_array_type
from web3._utils.validation import validate_abi_type, validate_abi_value
from web3.datastructures import AttributeDict

def hex_encode_abi_type(abi_type: TypeStr, value: Any, force_size: Optional[int]=None) -> HexStr:
    """
    Encodes value into a hex string in format of abi_type
    """
    validate_abi_type(abi_type)
    validate_abi_value(abi_type, value)

    data_size = force_size or size_of_type(abi_type)
    if is_array_type(abi_type):
        sub_type = sub_type_of_array_type(abi_type)
        return HexStr(''.join(
            remove_0x_prefix(hex_encode_abi_type(sub_type, v, force_size))
            for v in value
        ))
    elif is_bool_type(abi_type):
        return to_hex_with_size(value, data_size)
    elif is_uint_type(abi_type):
        return to_hex_with_size(value, data_size)
    elif is_int_type(abi_type):
        return to_hex_twos_compliment(value, data_size)
    elif is_address_type(abi_type):
        return pad_hex(value, data_size)
    elif is_bytes_type(abi_type):
        if is_bytes(value):
            return encode_hex(value)
        else:
            return value
    elif is_string_type(abi_type):
        return encode_hex(text_if_str(to_bytes, value))
    else:
        raise ValueError(f"Unsupported ABI type: {abi_type}")

def to_hex_twos_compliment(value: Any, bit_size: int) -> HexStr:
    """
    Converts integer value to twos compliment hex representation with given bit_size
    """
    if value >= 0:
        return to_hex_with_size(value, bit_size)

    value = (1 << bit_size) + value
    hex_value = hex(value)[2:]
    hex_size = bit_size // 4

    return add_0x_prefix(hex_value.zfill(hex_size))

def to_hex_with_size(value: Any, bit_size: int) -> HexStr:
    """
    Converts a value to hex with given bit_size:
    """
    if not is_list_like(value):
        value = [value]

    hex_value = encode_hex(value[0])[2:]
    hex_size = bit_size // 4

    return add_0x_prefix(hex_value.zfill(hex_size))

def pad_hex(value: Any, bit_size: int) -> HexStr:
    """
    Pads a hex string up to the given bit_size
    """
    value = remove_0x_prefix(HexStr(value))
    return add_0x_prefix(value.zfill(bit_size // 4))
zpad_bytes = pad_bytes(b'\x00')

@curry
def text_if_str(to_type: Callable[..., str], text_or_primitive: Union[Primitives, HexStr, str]) -> str:
    """
    Convert to a type, assuming that strings can be only unicode text (not a hexstr)

    @param to_type is a function that takes the arguments (primitive, hexstr=hexstr,
        text=text), eg~ to_bytes, to_text, to_hex, to_int, etc
    @param text_or_primitive in bytes, str, or int.
    """
    if isinstance(text_or_primitive, (bytes, int, bool)):
        return to_type(text_or_primitive)
    elif isinstance(text_or_primitive, str):
        return to_type(text_or_primitive, text=text_or_primitive)
    else:
        raise TypeError(
            "Expected string, bytes, int, or bool. Got {}".format(type(text_or_primitive))
        )

@curry
def hexstr_if_str(to_type: Callable[..., HexStr], hexstr_or_primitive: Union[Primitives, HexStr, str]) -> HexStr:
    """
    Convert to a type, assuming that strings can be only hexstr (not unicode text)

    @param to_type is a function that takes the arguments (primitive, hexstr=hexstr,
        text=text), eg~ to_bytes, to_text, to_hex, to_int, etc
    @param hexstr_or_primitive in bytes, str, or int.
    """
    if isinstance(hexstr_or_primitive, (bytes, int, bool)):
        return to_type(hexstr_or_primitive)
    elif isinstance(hexstr_or_primitive, str):
        if is_hex(hexstr_or_primitive):
            return to_type(hexstr_or_primitive, hexstr=hexstr_or_primitive)
        else:
            raise ValueError(
                "When the type is 'hexstr' the value must be a valid hex string. "
                "Got: {}".format(hexstr_or_primitive)
            )
    else:
        raise TypeError(
            "Expected string, bytes, int, or bool. Got {}".format(type(hexstr_or_primitive))
        )

class FriendlyJsonSerde:
    """
    Friendly JSON serializer & deserializer

    When encoding or decoding fails, this class collects
    information on which fields failed, to show more
    helpful information in the raised error messages.
    """
    def _json_mapping_errors(self, mapping: Dict[Any, Any], field_path: str='') -> Iterable[str]:
        for key, val in mapping.items():
            try:
                json.dumps(key)
                json.dumps(val)
            except TypeError as exc:
                field_name = field_path + str(key)
                yield f"{field_name}: {exc}"
                if isinstance(val, dict):
                    yield from self._json_mapping_errors(val, field_name + '.')

    def json_decode(self, json_str: str) -> Dict[Any, Any]:
        try:
            decoded = json.loads(json_str)
        except json.decoder.JSONDecodeError as exc:
            raise ValueError(f"Could not decode json: {exc}")
        return decoded

    def json_encode(self, value: Dict[Any, Any], cls: Optional[Type[json.JSONEncoder]]=None) -> str:
        try:
            return json.dumps(value, cls=cls)
        except TypeError as exc:
            if json.dumps([]) == '[]':
                # TypeError not caused by json module, let it bubble up
                raise
            # Get information about which fields failed
            mapping_errors = '\n'.join(self._json_mapping_errors(value))
            raise TypeError(f"Could not encode to JSON: {exc}\nMapping errors: {mapping_errors}")

class DynamicArrayPackedEncoder(BaseArrayEncoder):
    is_dynamic = True

class Web3JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, AttributeDict):
            return {key: value for key, value in obj.items()}
        if isinstance(obj, HexBytes):
            return obj.hex()
        return json.JSONEncoder.default(self, obj)

def to_json(obj: Dict[Any, Any]) -> str:
    """
    Convert a complex object (like a transaction object) to a JSON string
    """
    return FriendlyJsonSerde().json_encode(obj, cls=Web3JsonEncoder)