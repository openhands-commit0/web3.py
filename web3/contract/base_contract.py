from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, Generator, Iterable, List, NoReturn, Optional, Sequence, Tuple, Type, Union, cast
import warnings
from eth_typing import Address, ChecksumAddress, HexStr
from eth_utils import add_0x_prefix, combomethod, encode_hex, function_abi_to_4byte_selector, is_list_like, is_text, to_tuple
from hexbytes import HexBytes
from web3._utils.abi import abi_to_signature, check_if_arguments_can_be_encoded, fallback_func_abi_exists, filter_by_type, get_constructor_abi, is_array_type, merge_args_and_kwargs, receive_func_abi_exists
from web3._utils.contracts import decode_transaction_data, encode_abi, find_matching_event_abi, find_matching_fn_abi, get_function_info, prepare_transaction
from web3._utils.datatypes import PropertyCheckingFactory
from web3._utils.decorators import deprecate_method
from web3._utils.empty import empty
from web3._utils.encoding import to_4byte_hex, to_hex
from web3._utils.events import AsyncEventFilterBuilder, EventFilterBuilder, get_event_data, is_dynamic_sized_type
from web3._utils.filters import construct_event_filter_params
from web3._utils.function_identifiers import FallbackFn, ReceiveFn
from web3._utils.normalizers import BASE_RETURN_NORMALIZERS
from web3.datastructures import AttributeDict, MutableAttributeDict
from web3.exceptions import ABIEventFunctionNotFound, ABIFunctionNotFound, FallbackNotFound, InvalidEventABI, LogTopicError, MismatchedABI, NoABIEventsFound, NoABIFound, NoABIFunctionsFound, Web3ValidationError
from web3.logs import DISCARD, IGNORE, STRICT, WARN, EventLogErrorFlags
from web3.types import ABI, ABIEvent, ABIFunction, BlockIdentifier, EventData, FilterParams, FunctionIdentifier, TContractFn, TxParams, TxReceipt
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3
    from .async_contract import AsyncContractFunction
    from .contract import ContractFunction

class BaseContractEvent:
    """Base class for contract events

    An event accessed via the api `contract.events.myEvents(*args, **kwargs)`
    is a subclass of this class.
    """
    address: ChecksumAddress = None
    event_name: str = None
    w3: Union['Web3', 'AsyncWeb3'] = None
    contract_abi: ABI = None
    abi: ABIEvent = None

    def __init__(self, *argument_names: Tuple[str]) -> None:
        if argument_names is None:
            self.argument_names = tuple()
        else:
            self.argument_names = argument_names
        self.abi = self._get_event_abi()

class BaseContractEvents:
    """Class containing contract event objects

    This is available via:

    .. code-block:: python

        >>> mycontract.events
        <web3.contract.ContractEvents object at 0x108afde10>

    To get list of all supported events in the contract ABI.
    This allows you to iterate over :class:`ContractEvent` proxy classes.

    .. code-block:: python

        >>> for e in mycontract.events: print(e)
        <class 'web3._utils.datatypes.LogAnonymous'>
        ...

    """

    def __init__(self, abi: ABI, w3: Union['Web3', 'AsyncWeb3'], contract_event_type: Type['BaseContractEvent'], address: Optional[ChecksumAddress]=None) -> None:
        if abi:
            self.abi = abi
            self._events = filter_by_type('event', self.abi)
            for event in self._events:
                setattr(self, event['name'], contract_event_type.factory(event['name'], w3=w3, contract_abi=self.abi, address=address, event_name=event['name']))

    def __getattr__(self, event_name: str) -> Type['BaseContractEvent']:
        if '_events' not in self.__dict__:
            raise NoABIEventsFound('The abi for this contract contains no event definitions. ', 'Are you sure you provided the correct contract abi?')
        elif event_name not in self.__dict__['_events']:
            raise ABIEventFunctionNotFound(f"The event '{event_name}' was not found in this contract's abi. ", 'Are you sure you provided the correct contract abi?')
        else:
            return super().__getattribute__(event_name)

    def __getitem__(self, event_name: str) -> Type['BaseContractEvent']:
        return getattr(self, event_name)

    def __iter__(self) -> Iterable[Type['BaseContractEvent']]:
        """Iterate over supported

        :return: Iterable of :class:`ContractEvent`
        """
        for event in self._events:
            yield self[event['name']]

    def __hasattr__(self, event_name: str) -> bool:
        try:
            return event_name in self.__dict__['_events']
        except ABIEventFunctionNotFound:
            return False

class BaseContractFunction:
    """Base class for contract functions

    A function accessed via the api `contract.functions.myMethod(*args, **kwargs)`
    is a subclass of this class.
    """
    address: ChecksumAddress = None
    function_identifier: FunctionIdentifier = None
    w3: Union['Web3', 'AsyncWeb3'] = None
    contract_abi: ABI = None
    abi: ABIFunction = None
    transaction: TxParams = None
    arguments: Tuple[Any, ...] = None
    decode_tuples: Optional[bool] = False
    args: Any = None
    kwargs: Any = None

    def __init__(self, abi: Optional[ABIFunction]=None) -> None:
        self.abi = abi
        self.fn_name = type(self).__name__
    _return_data_normalizers: Optional[Tuple[Callable[..., Any], ...]] = tuple()

    def __repr__(self) -> str:
        if self.abi:
            _repr = f'<Function {abi_to_signature(self.abi)}'
            if self.arguments is not None:
                _repr += f' bound to {self.arguments!r}'
            return _repr + '>'
        return f'<Function {self.fn_name}>'

class BaseContractFunctions:
    """Class containing contract function objects"""

    def __init__(self, abi: ABI, w3: Union['Web3', 'AsyncWeb3'], contract_function_class: Union[Type['ContractFunction'], Type['AsyncContractFunction']], address: Optional[ChecksumAddress]=None, decode_tuples: Optional[bool]=False) -> None:
        self.abi = abi
        self.w3 = w3
        self.address = address
        if self.abi:
            self._functions = filter_by_type('function', self.abi)
            for func in self._functions:
                setattr(self, func['name'], contract_function_class.factory(func['name'], w3=self.w3, contract_abi=self.abi, address=self.address, decode_tuples=decode_tuples, function_identifier=func['name']))

    def __iter__(self) -> Generator[str, None, None]:
        if not hasattr(self, '_functions') or not self._functions:
            return
        for func in self._functions:
            yield func['name']

    def __getitem__(self, function_name: str) -> ABIFunction:
        return getattr(self, function_name)

    def __hasattr__(self, function_name: str) -> bool:
        try:
            return function_name in self.__dict__['_functions']
        except ABIFunctionNotFound:
            return False

class BaseContract:
    """Base class for Contract proxy classes.

    First you need to create your Contract classes using
    :meth:`web3.eth.Eth.contract` that takes compiled Solidity contract
    ABI definitions as input.  The created class object will be a subclass of
    this base class.

    After you have your Contract proxy class created you can interact with
    smart contracts

    * Create a Contract proxy object for an existing deployed smart contract by
      its address using :meth:`__init__`

    * Deploy a new smart contract using :py:meth:`Contract.constructor.transact()`
    """
    w3: Union['Web3', 'AsyncWeb3'] = None
    address: ChecksumAddress = None
    abi: ABI = None
    asm = None
    ast = None
    bytecode = None
    bytecode_runtime = None
    clone_bin = None
    decode_tuples = None
    dev_doc = None
    interface = None
    metadata = None
    opcodes = None
    src_map = None
    src_map_runtime = None
    user_doc = None

    @combomethod
    def encode_abi(cls, fn_name: str, args: Optional[Any]=None, kwargs: Optional[Any]=None, data: Optional[HexStr]=None) -> HexStr:
        """
        Encodes the arguments using the Ethereum ABI for the contract function
        that matches the given name and arguments..

        :param data: defaults to function selector
        """
        pass
    _return_data_normalizers: Tuple[Callable[..., Any], ...] = tuple()

class BaseContractCaller:
    """
    An alternative Contract API.

    This call:

    > contract.caller({'from': eth.accounts[1], 'gas': 100000, ...}).add(2, 3)
    is equivalent to this call in the classic contract:
    > contract.functions.add(2, 3).call({'from': eth.accounts[1], 'gas': 100000, ...})

    Other options for invoking this class include:

    > contract.caller.add(2, 3)

    or

    > contract.caller().add(2, 3)

    or

    > contract.caller(transaction={'from': eth.accounts[1], 'gas': 100000, ...}).add(2, 3)  # noqa: E501
    """
    _functions: List[Union[ABIFunction, ABIEvent]]

    def __init__(self, abi: ABI, w3: Union['Web3', 'AsyncWeb3'], address: ChecksumAddress, decode_tuples: Optional[bool]=False) -> None:
        self.w3 = w3
        self.address = address
        self.abi = abi
        self.decode_tuples = decode_tuples
        self._functions = []

    def __getattr__(self, function_name: str) -> Any:
        if self.abi is None:
            raise NoABIFound('There is no ABI found for this contract.')
        elif not self._functions or len(self._functions) == 0:
            raise NoABIFunctionsFound('The ABI for this contract contains no function definitions. ', 'Are you sure you provided the correct contract ABI?')
        elif function_name not in {fn['name'] for fn in self._functions}:
            functions_available = ', '.join([fn['name'] for fn in self._functions])
            raise ABIFunctionNotFound(f"The function '{function_name}' was not found in this contract's ABI.", ' Here is a list of all of the function names found: ', f'{functions_available}. ', 'Did you mean to call one of those functions?')
        else:
            return super().__getattribute__(function_name)

    def __hasattr__(self, event_name: str) -> bool:
        try:
            return event_name in self.__dict__['_events']
        except ABIFunctionNotFound:
            return False

class BaseContractConstructor:
    """
    Class for contract constructor API.
    """

    def __init__(self, w3: Union['Web3', 'AsyncWeb3'], abi: ABI, bytecode: HexStr, *args: Any, **kwargs: Any) -> None:
        self.w3 = w3
        self.abi = abi
        self.bytecode = bytecode
        self.data_in_transaction = self._encode_data_in_transaction(*args, **kwargs)

class NonExistentFallbackFunction:

    def __getattr__(self, attr: Any) -> Callable[[], None]:
        return self._raise_exception

class NonExistentReceiveFunction:

    def __getattr__(self, attr: Any) -> Callable[[], None]:
        return self._raise_exception