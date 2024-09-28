import copy
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Type, cast
from eth_typing import ChecksumAddress
from eth_utils import combomethod
from eth_utils.toolz import partial
from hexbytes import HexBytes
from web3._utils.abi import fallback_func_abi_exists, filter_by_type, receive_func_abi_exists
from web3._utils.async_transactions import async_fill_transaction_defaults
from web3._utils.compat import Self
from web3._utils.contracts import async_parse_block_identifier, parse_block_identifier_no_extra_call
from web3._utils.datatypes import PropertyCheckingFactory
from web3._utils.events import AsyncEventFilterBuilder, get_event_data
from web3._utils.filters import AsyncLogFilter
from web3._utils.function_identifiers import FallbackFn, ReceiveFn
from web3._utils.normalizers import normalize_abi, normalize_address_no_ens, normalize_bytecode
from web3.contract.base_contract import BaseContract, BaseContractCaller, BaseContractConstructor, BaseContractEvent, BaseContractEvents, BaseContractFunction, BaseContractFunctions, NonExistentFallbackFunction, NonExistentReceiveFunction
from web3.contract.utils import async_build_transaction_for_function, async_call_contract_function, async_estimate_gas_for_function, async_transact_with_contract_function, find_functions_by_identifier, get_function_by_identifier
from web3.exceptions import ABIFunctionNotFound, NoABIFound, NoABIFunctionsFound, Web3ValidationError
from web3.types import ABI, BlockIdentifier, CallOverride, EventData, TxParams
from web3.utils import get_abi_input_names
if TYPE_CHECKING:
    from ens import AsyncENS
    from web3 import AsyncWeb3

class AsyncContractEvent(BaseContractEvent):
    w3: 'AsyncWeb3'

    @combomethod
    async def get_logs(self, argument_filters: Optional[Dict[str, Any]]=None, fromBlock: Optional[BlockIdentifier]=None, toBlock: Optional[BlockIdentifier]=None, block_hash: Optional[HexBytes]=None) -> Awaitable[Iterable[EventData]]:
        """Get events for this contract instance using eth_getLogs API.

        This is a stateless method, as opposed to createFilter.
        It can be safely called against nodes which do not provide
        eth_newFilter API, like Infura nodes.

        If there are many events,
        like ``Transfer`` events for a popular token,
        the Ethereum node might be overloaded and timeout
        on the underlying JSON-RPC call.

        Example - how to get all ERC-20 token transactions
        for the latest 10 blocks:

        .. code-block:: python

            from = max(mycontract.web3.eth.block_number - 10, 1)
            to = mycontract.web3.eth.block_number

            events = mycontract.events.Transfer.getLogs(fromBlock=from, toBlock=to)

            for e in events:
                print(e["args"]["from"],
                    e["args"]["to"],
                    e["args"]["value"])

        The returned processed log values will look like:

        .. code-block:: python

            (
                AttributeDict({
                 'args': AttributeDict({}),
                 'event': 'LogNoArguments',
                 'logIndex': 0,
                 'transactionIndex': 0,
                 'transactionHash': HexBytes('...'),
                 'address': '0xF2E246BB76DF876Cef8b38ae84130F4F55De395b',
                 'blockHash': HexBytes('...'),
                 'blockNumber': 3
                }),
                AttributeDict(...),
                ...
            )

        See also: :func:`web3.middleware.filter.local_filter_middleware`.

        :param argument_filters: Filter by argument values. Indexed arguments are
          filtered by the node while non-indexed arguments are filtered by the library.
        :param fromBlock: block number or "latest", defaults to "latest"
        :param toBlock: block number or "latest". Defaults to "latest"
        :param block_hash: block hash. Cannot be set at the
          same time as fromBlock or toBlock
        :yield: Tuple of :class:`AttributeDict` instances
        """
        pass

    @combomethod
    async def create_filter(self, *, argument_filters: Optional[Dict[str, Any]]=None, fromBlock: Optional[BlockIdentifier]=None, toBlock: BlockIdentifier='latest', address: Optional[ChecksumAddress]=None, topics: Optional[Sequence[Any]]=None) -> AsyncLogFilter:
        """
        Create filter object that tracks logs emitted by this contract event.
        """
        pass

class AsyncContractEvents(BaseContractEvents):

    def __init__(self, abi: ABI, w3: 'AsyncWeb3', address: Optional[ChecksumAddress]=None) -> None:
        super().__init__(abi, w3, AsyncContractEvent, address)

class AsyncContractFunction(BaseContractFunction):
    w3: 'AsyncWeb3'

    def __call__(self, *args: Any, **kwargs: Any) -> 'AsyncContractFunction':
        clone = copy.copy(self)
        if args is None:
            clone.args = tuple()
        else:
            clone.args = args
        if kwargs is None:
            clone.kwargs = {}
        else:
            clone.kwargs = kwargs
        clone._set_function_info()
        return clone

    async def call(self, transaction: Optional[TxParams]=None, block_identifier: BlockIdentifier=None, state_override: Optional[CallOverride]=None, ccip_read_enabled: Optional[bool]=None) -> Any:
        """
        Execute a contract function call using the `eth_call` interface.

        This method prepares a ``Caller`` object that exposes the contract
        functions and public variables as callable Python functions.

        Reading a public ``owner`` address variable example:

        .. code-block:: python

            ContractFactory = w3.eth.contract(
                abi=wallet_contract_definition["abi"]
            )

            # Not a real contract address
            contract = ContractFactory("0x2f70d3d26829e412A602E83FE8EeBF80255AEeA5")

            # Read "owner" public variable
            addr = contract.functions.owner().call()

        :param transaction: Dictionary of transaction info for web3 interface
        :param block_identifier TODO
        :param state_override TODO
        :param ccip_read_enabled TODO
        :return: ``Caller`` object that has contract public functions
            and variables exposed as Python methods
        """
        pass

class AsyncContractFunctions(BaseContractFunctions):

    def __init__(self, abi: ABI, w3: 'AsyncWeb3', address: Optional[ChecksumAddress]=None, decode_tuples: Optional[bool]=False) -> None:
        super().__init__(abi, w3, AsyncContractFunction, address, decode_tuples)

    def __getattr__(self, function_name: str) -> 'AsyncContractFunction':
        if self.abi is None:
            raise NoABIFound('There is no ABI found for this contract.')
        if '_functions' not in self.__dict__:
            raise NoABIFunctionsFound('The abi for this contract contains no function definitions. ', 'Are you sure you provided the correct contract abi?')
        elif function_name not in self.__dict__['_functions']:
            raise ABIFunctionNotFound(f"The function '{function_name}' was not found in this contract's abi.", ' Are you sure you provided the correct contract abi?')
        else:
            return super().__getattribute__(function_name)

class AsyncContract(BaseContract):
    functions: AsyncContractFunctions = None
    caller: 'AsyncContractCaller' = None
    w3: 'AsyncWeb3'
    events: AsyncContractEvents = None

    def __init__(self, address: Optional[ChecksumAddress]=None) -> None:
        """Create a new smart contract proxy object.

        :param address: Contract address as 0x hex string"""
        if self.w3 is None:
            raise AttributeError('The `Contract` class has not been initialized.  Please use the `web3.contract` interface to create your contract class.')
        if address:
            self.address = normalize_address_no_ens(address)
        if not self.address:
            raise TypeError('The address argument is required to instantiate a contract.')
        self.functions = AsyncContractFunctions(self.abi, self.w3, self.address, decode_tuples=self.decode_tuples)
        self.caller = AsyncContractCaller(self.abi, self.w3, self.address, decode_tuples=self.decode_tuples)
        self.events = AsyncContractEvents(self.abi, self.w3, self.address)
        self.fallback = AsyncContract.get_fallback_function(self.abi, self.w3, AsyncContractFunction, self.address)
        self.receive = AsyncContract.get_receive_function(self.abi, self.w3, AsyncContractFunction, self.address)

    @classmethod
    def constructor(cls, *args: Any, **kwargs: Any) -> Self:
        """
        :param args: The contract constructor arguments as positional arguments
        :param kwargs: The contract constructor arguments as keyword arguments
        :return: a contract constructor object
        """
        pass

class AsyncContractCaller(BaseContractCaller):
    w3: 'AsyncWeb3'

    def __init__(self, abi: ABI, w3: 'AsyncWeb3', address: ChecksumAddress, transaction: Optional[TxParams]=None, block_identifier: BlockIdentifier=None, ccip_read_enabled: Optional[bool]=None, decode_tuples: Optional[bool]=False) -> None:
        super().__init__(abi, w3, address, decode_tuples=decode_tuples)
        if self.abi:
            if transaction is None:
                transaction = {}
            self._functions = filter_by_type('function', self.abi)
            for func in self._functions:
                fn = AsyncContractFunction.factory(func['name'], w3=w3, contract_abi=self.abi, address=self.address, function_identifier=func['name'], decode_tuples=decode_tuples)
                block_id = parse_block_identifier_no_extra_call(w3, block_identifier)
                caller_method = partial(self.call_function, fn, transaction=transaction, block_identifier=block_id, ccip_read_enabled=ccip_read_enabled)
                setattr(self, func['name'], caller_method)

    def __call__(self, transaction: Optional[TxParams]=None, block_identifier: BlockIdentifier=None, ccip_read_enabled: Optional[bool]=None) -> 'AsyncContractCaller':
        if transaction is None:
            transaction = {}
        return type(self)(self.abi, self.w3, self.address, transaction=transaction, block_identifier=block_identifier, ccip_read_enabled=ccip_read_enabled, decode_tuples=self.decode_tuples)

class AsyncContractConstructor(BaseContractConstructor):
    w3: 'AsyncWeb3'

    @combomethod
    async def build_transaction(self, transaction: Optional[TxParams]=None) -> TxParams:
        """
        Build the transaction dictionary without sending
        """
        pass