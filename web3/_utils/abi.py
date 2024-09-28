import binascii
from collections import abc, namedtuple
import copy
import itertools
import re
from typing import TYPE_CHECKING, Any, Callable, Collection, Coroutine, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast
from eth_abi import codec, decoding, encoding
from eth_abi.base import parse_type_str
from eth_abi.exceptions import ValueOutOfBounds
from eth_abi.grammar import ABIType, BasicType, TupleType, parse
from eth_abi.registry import ABIRegistry, BaseEquals, registry as default_registry
from eth_typing import HexStr, TypeStr
from eth_utils import decode_hex, is_bytes, is_list_like, is_string, is_text, to_text, to_tuple
from eth_utils.abi import collapse_if_tuple
from eth_utils.toolz import curry, partial, pipe
from web3._utils.decorators import reject_recursive_repeats
from web3._utils.ens import is_ens_name
from web3._utils.formatters import recursive_map
from web3.exceptions import FallbackNotFound, MismatchedABI
from web3.types import ABI, ABIEvent, ABIEventParams, ABIFunction, ABIFunctionParams, TReturn
from web3.utils import get_abi_input_names
if TYPE_CHECKING:
    from web3 import AsyncWeb3

def get_normalized_abi_arg_type(abi_arg: ABIEventParams) -> str:
    """
    Return the normalized type for the abi argument provided.
    In order to account for tuple argument types, this abstraction
    makes use of `collapse_if_tuple()` to collapse the appropriate component
    types within a tuple type, if present.
    """
    pass

class AddressEncoder(encoding.AddressEncoder):
    pass

class AcceptsHexStrEncoder(encoding.BaseEncoder):
    subencoder_cls: Type[encoding.BaseEncoder] = None
    is_strict: bool = None
    is_big_endian: bool = False
    data_byte_size: int = None
    value_bit_size: int = None

    def __init__(self, subencoder: encoding.BaseEncoder, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.subencoder = subencoder
        self.is_dynamic = subencoder.is_dynamic

class BytesEncoder(AcceptsHexStrEncoder):
    subencoder_cls = encoding.BytesEncoder
    is_strict = False

class ExactLengthBytesEncoder(BytesEncoder):
    is_strict = True

class ByteStringEncoder(AcceptsHexStrEncoder):
    subencoder_cls = encoding.ByteStringEncoder
    is_strict = False

class StrictByteStringEncoder(AcceptsHexStrEncoder):
    subencoder_cls = encoding.ByteStringEncoder
    is_strict = True

class TextStringEncoder(encoding.TextStringEncoder):
    pass

def merge_args_and_kwargs(function_abi: ABIFunction, args: Sequence[Any], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Takes a list of positional args (``args``) and a dict of keyword args
    (``kwargs``) defining values to be passed to a call to the contract function
    described by ``function_abi``.  Checks to ensure that the correct number of
    args were given, no duplicate args were given, and no unknown args were
    given.  Returns a list of argument values aligned to the order of inputs
    defined in ``function_abi``.
    """
    pass
TUPLE_TYPE_STR_RE = re.compile('^(tuple)((\\[([1-9]\\d*\\b)?])*)??$')

def get_tuple_type_str_parts(s: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Takes a JSON ABI type string.  For tuple type strings, returns the separated
    prefix and array dimension parts.  For all other strings, returns ``None``.
    """
    pass

def _align_abi_input(arg_abi: ABIFunctionParams, arg: Any) -> Tuple[Any, ...]:
    """
    Aligns the values of any mapping at any level of nesting in ``arg``
    according to the layout of the corresponding abi spec.
    """
    pass

def get_aligned_abi_inputs(abi: ABIFunction, args: Union[Tuple[Any, ...], Mapping[Any, Any]]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """
    Takes a function ABI (``abi``) and a sequence or mapping of args (``args``).
    Returns a list of type strings for the function's inputs and a list of
    arguments which have been aligned to the layout of those types.  The args
    contained in ``args`` may contain nested mappings or sequences corresponding
    to tuple-encoded values in ``abi``.
    """
    pass
DYNAMIC_TYPES = ['bytes', 'string']
INT_SIZES = range(8, 257, 8)
BYTES_SIZES = range(1, 33)
UINT_TYPES = [f'uint{i}' for i in INT_SIZES]
INT_TYPES = [f'int{i}' for i in INT_SIZES]
BYTES_TYPES = [f'bytes{i}' for i in BYTES_SIZES] + ['bytes32.byte']
STATIC_TYPES = list(itertools.chain(['address', 'bool'], UINT_TYPES, INT_TYPES, BYTES_TYPES))
BASE_TYPE_REGEX = '|'.join((_type + '(?![a-z0-9])' for _type in itertools.chain(STATIC_TYPES, DYNAMIC_TYPES)))
SUB_TYPE_REGEX = '\\[[0-9]*\\]'
TYPE_REGEX = '^(?:{base_type})(?:(?:{sub_type})*)?$'.format(base_type=BASE_TYPE_REGEX, sub_type=SUB_TYPE_REGEX)

def size_of_type(abi_type: TypeStr) -> int:
    """
    Returns size in bits of abi_type
    """
    pass
END_BRACKETS_OF_ARRAY_TYPE_REGEX = '\\[[^]]*\\]$'
ARRAY_REGEX = '^[a-zA-Z0-9_]+({sub_type})+$'.format(sub_type=SUB_TYPE_REGEX)
NAME_REGEX = '[a-zA-Z_][a-zA-Z0-9_]*'
ENUM_REGEX = '^{lib_name}\\.{enum_name}$'.format(lib_name=NAME_REGEX, enum_name=NAME_REGEX)

@curry
def map_abi_data(normalizers: Sequence[Callable[[TypeStr, Any], Tuple[TypeStr, Any]]], types: Sequence[TypeStr], data: Sequence[Any]) -> Any:
    """
    This function will apply normalizers to your data, in the
    context of the relevant types. Each normalizer is in the format:

    def normalizer(datatype, data):
        # Conditionally modify data
        return (datatype, data)

    Where datatype is a valid ABI type string, like "uint".

    In case of an array, like "bool[2]", normalizer will receive `data`
    as an iterable of typed data, like `[("bool", True), ("bool", False)]`.

    Internals
    ---

    This is accomplished by:

    1. Decorating the data tree with types
    2. Recursively mapping each of the normalizers to the data
    3. Stripping the types back out of the tree
    """
    pass

@curry
def abi_data_tree(types: Sequence[TypeStr], data: Sequence[Any]) -> List[Any]:
    """
    Decorate the data tree with pairs of (type, data). The pair tuple is actually an
    ABITypedData, but can be accessed as a tuple.

    As an example:

    >>> abi_data_tree(types=["bool[2]", "uint"], data=[[True, False], 0])
    [("bool[2]", [("bool", True), ("bool", False)]), ("uint256", 0)]
    """
    pass

@curry
def data_tree_map(func: Callable[[TypeStr, Any], Tuple[TypeStr, Any]], data_tree: Any) -> 'ABITypedData':
    """
    Map func to every ABITypedData element in the tree. func will
    receive two args: abi_type, and data
    """
    pass

class ABITypedData(namedtuple('ABITypedData', 'abi_type, data')):
    """
    This class marks data as having a certain ABI-type.

    >>> a1 = ABITypedData(['address', addr1])
    >>> a2 = ABITypedData(['address', addr2])
    >>> addrs = ABITypedData(['address[]', [a1, a2]])

    You can access the fields using tuple() interface, or with
    attributes:

    >>> assert a1.abi_type == a1[0]
    >>> assert a1.data == a1[1]

    Unlike a typical `namedtuple`, you initialize with a single
    positional argument that is iterable, to match the init
    interface of all other relevant collections.
    """

    def __new__(cls, iterable: Iterable[Any]) -> 'ABITypedData':
        return super().__new__(cls, *iterable)

def named_tree(abi: Iterable[Union[ABIFunctionParams, ABIFunction, ABIEvent, Dict[TypeStr, Any]]], data: Iterable[Tuple[Any, ...]]) -> Dict[str, Any]:
    """
    Convert function inputs/outputs or event data tuple to dict with names from ABI.
    """
    pass

async def async_data_tree_map(async_w3: 'AsyncWeb3', func: Callable[['AsyncWeb3', TypeStr, Any], Coroutine[Any, Any, Tuple[TypeStr, Any]]], data_tree: Any) -> 'ABITypedData':
    """
    Map an awaitable method to every ABITypedData element in the tree.

    The awaitable method should receive three positional args:
        async_w3, abi_type, and data
    """
    pass

@reject_recursive_repeats
async def async_recursive_map(async_w3: 'AsyncWeb3', func: Callable[[Any], Coroutine[Any, Any, TReturn]], data: Any) -> TReturn:
    """
    Apply an awaitable method to data and any collection items inside data
    (using async_map_collection).

    Define the awaitable method so that it only applies to the type of value that you
    want it to apply to.
    """
    pass

async def async_map_if_collection(func: Callable[[Any], Coroutine[Any, Any, Any]], value: Any) -> Any:
    """
    Apply an awaitable method to each element of a collection or value of a dictionary.
    If the value is not a collection, return it unmodified.
    """
    pass