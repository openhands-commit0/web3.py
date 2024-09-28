import itertools
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, Type, Union
from eth_abi.exceptions import DecodingError
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from web3._utils.abi import filter_by_type, get_abi_output_types, map_abi_data, named_tree, recursive_dict_to_namedtuple
from web3._utils.async_transactions import async_fill_transaction_defaults
from web3._utils.contracts import find_matching_fn_abi, prepare_transaction
from web3._utils.normalizers import BASE_RETURN_NORMALIZERS
from web3._utils.transactions import fill_transaction_defaults
from web3.exceptions import BadFunctionCallOutput
from web3.types import ABI, ABIFunction, BlockIdentifier, CallOverride, FunctionIdentifier, TContractFn, TxParams
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3
ACCEPTABLE_EMPTY_STRINGS = ['0x', b'0x', '', b'']

def call_contract_function(w3: 'Web3', address: ChecksumAddress, normalizers: Tuple[Callable[..., Any], ...], function_identifier: FunctionIdentifier, transaction: TxParams, block_id: Optional[BlockIdentifier]=None, contract_abi: Optional[ABI]=None, fn_abi: Optional[ABIFunction]=None, state_override: Optional[CallOverride]=None, ccip_read_enabled: Optional[bool]=None, decode_tuples: Optional[bool]=False, *args: Any, **kwargs: Any) -> Any:
    """
    Helper function for interacting with a contract function using the
    `eth_call` API.
    """
    pass

def transact_with_contract_function(address: ChecksumAddress, w3: 'Web3', function_name: Optional[FunctionIdentifier]=None, transaction: Optional[TxParams]=None, contract_abi: Optional[ABI]=None, fn_abi: Optional[ABIFunction]=None, *args: Any, **kwargs: Any) -> HexBytes:
    """
    Helper function for interacting with a contract function by sending a
    transaction.
    """
    pass

def estimate_gas_for_function(address: ChecksumAddress, w3: 'Web3', fn_identifier: Optional[FunctionIdentifier]=None, transaction: Optional[TxParams]=None, contract_abi: Optional[ABI]=None, fn_abi: Optional[ABIFunction]=None, block_identifier: Optional[BlockIdentifier]=None, state_override: Optional[CallOverride]=None, *args: Any, **kwargs: Any) -> int:
    """Estimates gas cost a function call would take.

    Don't call this directly, instead use :meth:`Contract.estimate_gas`
    on your contract instance.
    """
    pass

def build_transaction_for_function(address: ChecksumAddress, w3: 'Web3', function_name: Optional[FunctionIdentifier]=None, transaction: Optional[TxParams]=None, contract_abi: Optional[ABI]=None, fn_abi: Optional[ABIFunction]=None, *args: Any, **kwargs: Any) -> TxParams:
    """Builds a dictionary with the fields required to make the given transaction

    Don't call this directly, instead use :meth:`Contract.build_transaction`
    on your contract instance.
    """
    pass

async def async_call_contract_function(async_w3: 'AsyncWeb3', address: ChecksumAddress, normalizers: Tuple[Callable[..., Any], ...], function_identifier: FunctionIdentifier, transaction: TxParams, block_id: Optional[BlockIdentifier]=None, contract_abi: Optional[ABI]=None, fn_abi: Optional[ABIFunction]=None, state_override: Optional[CallOverride]=None, ccip_read_enabled: Optional[bool]=None, decode_tuples: Optional[bool]=False, *args: Any, **kwargs: Any) -> Any:
    """
    Helper function for interacting with a contract function using the
    `eth_call` API.
    """
    pass

async def async_transact_with_contract_function(address: ChecksumAddress, async_w3: 'AsyncWeb3', function_name: Optional[FunctionIdentifier]=None, transaction: Optional[TxParams]=None, contract_abi: Optional[ABI]=None, fn_abi: Optional[ABIFunction]=None, *args: Any, **kwargs: Any) -> HexBytes:
    """
    Helper function for interacting with a contract function by sending a
    transaction.
    """
    pass

async def async_estimate_gas_for_function(address: ChecksumAddress, async_w3: 'AsyncWeb3', fn_identifier: Optional[FunctionIdentifier]=None, transaction: Optional[TxParams]=None, contract_abi: Optional[ABI]=None, fn_abi: Optional[ABIFunction]=None, block_identifier: Optional[BlockIdentifier]=None, state_override: Optional[CallOverride]=None, *args: Any, **kwargs: Any) -> int:
    """Estimates gas cost a function call would take.

    Don't call this directly, instead use :meth:`Contract.estimate_gas`
    on your contract instance.
    """
    pass

async def async_build_transaction_for_function(address: ChecksumAddress, async_w3: 'AsyncWeb3', function_name: Optional[FunctionIdentifier]=None, transaction: Optional[TxParams]=None, contract_abi: Optional[ABI]=None, fn_abi: Optional[ABIFunction]=None, *args: Any, **kwargs: Any) -> TxParams:
    """Builds a dictionary with the fields required to make the given transaction

    Don't call this directly, instead use :meth:`Contract.build_transaction`
    on your contract instance.
    """
    pass