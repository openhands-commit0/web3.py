from typing import Any
from eth_utils import is_0x_prefixed, is_hex_address

def is_ens_name(value: Any) -> bool:
    """
    Check if the given value is a valid ENS name.
    
    A valid ENS name:
    - Is a string
    - Contains at least one dot (.)
    - Is not a hex address
    - Is not 0x-prefixed
    """
    if not isinstance(value, str):
        return False
    if not "." in value:
        return False
    if is_hex_address(value):
        return False
    if is_0x_prefixed(value):
        return False
    return True