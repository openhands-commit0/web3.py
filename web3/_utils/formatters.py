from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, Tuple, TypeVar
from eth_typing import HexStr
from eth_utils import is_dict, is_list_like, is_string, to_dict
from eth_utils.curried import apply_formatter_at_index
from eth_utils.toolz import compose, curry, dissoc
from web3._utils.decorators import reject_recursive_repeats
TReturn = TypeVar('TReturn')
TValue = TypeVar('TValue')
integer_to_hex = hex

def map_collection(func: Callable[..., TReturn], collection: Any) -> Any:
    """
    Apply func to each element of a collection, or value of a dictionary.
    If the value is not a collection, return it unmodified
    """
    pass

@reject_recursive_repeats
def recursive_map(func: Callable[..., TReturn], data: Any) -> TReturn:
    """
    Apply func to data, and any collection items inside data (using map_collection).
    Define func so that it only applies to the type of value that you
    want it to apply to.
    """
    pass