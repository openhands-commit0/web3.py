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
    if isinstance(abi_arg['type'], str):
        return collapse_if_tuple(dict(abi_arg))
    raise ValueError(f"Unknown ABI arg type: {abi_arg['type']}")

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
    if not function_abi.get('inputs', None):
        if args or kwargs:
            raise TypeError(
                "Function {} does not accept any arguments".format(function_abi.get('name', '[fallback]'))
            )
        return ()

    input_names = get_abi_input_names(function_abi)
    num_inputs = len(input_names)

    # Check that all positional args are in bounds
    if len(args) > num_inputs:
        raise TypeError(
            "Function {} received too many positional values".format(function_abi.get('name', '[fallback]'))
        )

    # Check that all keyword args are known
    for key in kwargs:
        if key not in input_names:
            raise TypeError(
                "{} is not a valid argument for function {}".format(
                    key, function_abi.get('name', '[fallback]')
                )
            )

    # Check that same argument is not passed twice
    for idx, value in enumerate(args):
        if input_names[idx] in kwargs:
            raise TypeError(
                "Function {} got multiple values for argument {}".format(
                    function_abi.get('name', '[fallback]'),
                    input_names[idx],
                )
            )

    # Fill remaining arguments with keyword values
    args_as_kwargs = {name: arg for name, arg in zip(input_names, args)}
    args_as_kwargs.update(kwargs)

    # Check that all required args have been given
    missing_args = set(input_names) - set(args_as_kwargs)
    if missing_args:
        raise TypeError(
            "Function {} missing arguments: {}".format(
                function_abi.get('name', '[fallback]'),
                ', '.join(missing_args),
            )
        )

    # Return values in order of inputs
    return tuple(args_as_kwargs[name] for name in input_names)
TUPLE_TYPE_STR_RE = re.compile('^(tuple)((\\[([1-9]\\d*\\b)?])*)??$')

def get_tuple_type_str_parts(s: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Takes a JSON ABI type string.  For tuple type strings, returns the separated
    prefix and array dimension parts.  For all other strings, returns ``None``.
    """
    match = TUPLE_TYPE_STR_RE.match(s)
    if match:
        tuple_prefix = match.group(1)
        array_part = match.group(2) or None
        return tuple_prefix, array_part
    return None

def _align_tuple_items(tuple_components: Sequence[ABIFunctionParams], tuple_value: Union[Sequence[Any], Mapping[Any, Any]]) -> Tuple[Any, ...]:
    """
    Takes a list of tuple component ABIs and a sequence or mapping of values.
    Returns a tuple of values aligned to the component ABIs.
    """
    if isinstance(tuple_value, Mapping):
        return tuple(
            _align_abi_input(comp_abi, tuple_value[comp_abi['name']])
            for comp_abi in tuple_components
        )
    
    return tuple(
        _align_abi_input(comp_abi, tuple_item)
        for comp_abi, tuple_item in zip(tuple_components, tuple_value)
    )

def _align_abi_input(arg_abi: ABIFunctionParams, arg: Any) -> Tuple[Any, ...]:
    """
    Aligns the values of any mapping at any level of nesting in ``arg``
    according to the layout of the corresponding abi spec.
    """
    tuple_parts = get_tuple_type_str_parts(arg_abi['type'])
    if tuple_parts is None:
        return arg

    tuple_prefix, array_part = tuple_parts
    tuple_components = arg_abi.get('components', None)
    if tuple_components is None:
        raise ValueError("Tuple components missing from ABI")

    if array_part:
        # If array dimension exists, map alignment to each tuple element
        if not is_list_like(arg):
            raise TypeError(
                "Expected list-like data for type {}, got {}".format(
                    arg_abi['type'], type(arg)
                )
            )
        return tuple(
            _align_tuple_items(tuple_components, tuple_item)
            for tuple_item in arg
        )

    return _align_tuple_items(tuple_components, arg)

def get_aligned_abi_inputs(abi: ABIFunction, args: Union[Tuple[Any, ...], Mapping[Any, Any]]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """
    Takes a function ABI (``abi``) and a sequence or mapping of args (``args``).
    Returns a list of type strings for the function's inputs and a list of
    arguments which have been aligned to the layout of those types.  The args
    contained in ``args`` may contain nested mappings or sequences corresponding
    to tuple-encoded values in ``abi``.
    """
    inputs = abi.get('inputs', [])
    if not inputs:
        if args and not isinstance(args, (tuple, list)) or (isinstance(args, (tuple, list)) and args):
            raise TypeError(
                "Function {} does not accept any arguments".format(abi.get('name', '[fallback]'))
            )
        return (), ()

    input_types = tuple(collapse_if_tuple(dict(arg)) for arg in inputs)

    if isinstance(args, (list, tuple)):
        if len(args) > len(inputs):
            raise TypeError(
                "Function {} received too many arguments".format(abi.get('name', '[fallback]'))
            )
        args_as_list = list(args)
    else:
        args_as_list = []
        for input_abi in inputs:
            if input_abi['name'] not in args:
                raise TypeError(
                    "Function {} missing argument: {}".format(
                        abi.get('name', '[fallback]'),
                        input_abi['name']
                    )
                )
            args_as_list.append(args[input_abi['name']])

    for i, input_abi in enumerate(inputs):
        if i < len(args_as_list):
            args_as_list[i] = _align_abi_input(input_abi, args_as_list[i])

    return input_types, tuple(args_as_list)
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
    if not is_recognized_type(abi_type):
        raise ValueError(f"Unrecognized abi_type: {abi_type}")
    
    if is_array_type(abi_type):
        sub_type = sub_type_of_array_type(abi_type)
        return size_of_type(sub_type)
    
    if abi_type == 'bool':
        return 8
    elif abi_type == 'address':
        return 160
    elif abi_type.startswith('bytes'):
        if abi_type == 'bytes':
            return None
        return int(abi_type[5:]) * 8
    elif abi_type.startswith('uint'):
        return int(abi_type[4:])
    elif abi_type.startswith('int'):
        return int(abi_type[3:])
    elif abi_type == 'string':
        return None
    
    raise ValueError(f"Unsupported abi_type: {abi_type}")
END_BRACKETS_OF_ARRAY_TYPE_REGEX = '\\[[^]]*\\]$'
ARRAY_REGEX = '^[a-zA-Z0-9_]+({sub_type})+$'.format(sub_type=SUB_TYPE_REGEX)
NAME_REGEX = '[a-zA-Z_][a-zA-Z0-9_]*'
ENUM_REGEX = '^{lib_name}\\.{enum_name}$'.format(lib_name=NAME_REGEX, enum_name=NAME_REGEX)

def _get_data(data_tree: Any) -> Any:
    """
    Extract data values from an ABITypedData tree.
    """
    if isinstance(data_tree, ABITypedData):
        return _get_data(data_tree.data)
    elif isinstance(data_tree, (list, tuple)):
        return type(data_tree)(_get_data(item) for item in data_tree)
    return data_tree

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
    pipeline = itertools.chain(
        [abi_data_tree(types)],
        map(data_tree_map, normalizers),
        [lambda tree: _get_data(tree)],
    )

    return pipe(data, *pipeline)

@curry
def abi_data_tree(types: Sequence[TypeStr], data: Sequence[Any]) -> List[Any]:
    """
    Decorate the data tree with pairs of (type, data). The pair tuple is actually an
    ABITypedData, but can be accessed as a tuple.

    As an example:

    >>> abi_data_tree(types=["bool[2]", "uint"], data=[[True, False], 0])
    [("bool[2]", [("bool", True), ("bool", False)]), ("uint256", 0)]
    """
    if len(types) != len(data):
        raise ValueError(
            "Length mismatch between types and data: got {0} types and {1} data items".format(
                len(types), len(data)
            )
        )

    results = []

    for data_type, data_value in zip(types, data):
        if is_array_type(data_type):
            item_type = sub_type_of_array_type(data_type)
            value_type = [
                abi_data_tree([item_type], [item])[0]
                for item in data_value
            ]
            results.append(ABITypedData([data_type, value_type]))
        else:
            results.append(ABITypedData([data_type, data_value]))

    return results

@curry
def data_tree_map(func: Callable[[TypeStr, Any], Tuple[TypeStr, Any]], data_tree: Any) -> 'ABITypedData':
    """
    Map func to every ABITypedData element in the tree. func will
    receive two args: abi_type, and data
    """
    if isinstance(data_tree, ABITypedData):
        abi_type, data = data_tree
        if is_array_type(abi_type):
            item_type = sub_type_of_array_type(abi_type)
            value_type = [
                data_tree_map(func, item)
                for item in data
            ]
            return ABITypedData(func(abi_type, value_type))
        else:
            return ABITypedData(func(abi_type, data))
    elif isinstance(data_tree, (list, tuple)):
        return type(data_tree)(data_tree_map(func, item) for item in data_tree)
    else:
        return data_tree

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
    names = [item['name'] for item in abi]
    items_with_name = [
        (name, data_item)
        for name, data_item
        in zip(names, data)
        if name
    ]

    return dict(items_with_name)

async def async_data_tree_map(async_w3: 'AsyncWeb3', func: Callable[['AsyncWeb3', TypeStr, Any], Coroutine[Any, Any, Tuple[TypeStr, Any]]], data_tree: Any) -> 'ABITypedData':
    """
    Map an awaitable method to every ABITypedData element in the tree.

    The awaitable method should receive three positional args:
        async_w3, abi_type, and data
    """
    if isinstance(data_tree, ABITypedData):
        abi_type, data = data_tree
        if is_array_type(abi_type):
            item_type = sub_type_of_array_type(abi_type)
            value_type = [
                await async_data_tree_map(async_w3, func, item)
                for item in data
            ]
            return ABITypedData(await func(async_w3, abi_type, value_type))
        else:
            return ABITypedData(await func(async_w3, abi_type, data))
    elif isinstance(data_tree, (list, tuple)):
        return type(data_tree)(
            await async_data_tree_map(async_w3, func, item)
            for item in data_tree
        )
    else:
        return data_tree

@reject_recursive_repeats
async def async_recursive_map(async_w3: 'AsyncWeb3', func: Callable[[Any], Coroutine[Any, Any, TReturn]], data: Any) -> TReturn:
    """
    Apply an awaitable method to data and any collection items inside data
    (using async_map_collection).

    Define the awaitable method so that it only applies to the type of value that you
    want it to apply to.
    """
    result = await func(data)
    return await async_map_if_collection(
        lambda item: async_recursive_map(async_w3, func, item),
        result
    )

async def async_map_if_collection(func: Callable[[Any], Coroutine[Any, Any, Any]], value: Any) -> Any:
    """
    Apply an awaitable method to each element of a collection or value of a dictionary.
    If the value is not a collection, return it unmodified.
    """
    if isinstance(value, dict):
        return {
            key: await async_map_if_collection(func, item)
            for key, item in value.items()
        }
    elif isinstance(value, (list, tuple)):
        return type(value)(
            await async_map_if_collection(func, item)
            for item in value
        )
    elif isinstance(value, abc.Collection) and not isinstance(value, (str, bytes)):
        return type(value)(
            await async_map_if_collection(func, item)
            for item in value
        )
    else:
        return value