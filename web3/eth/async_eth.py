import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union, cast, overload
import warnings
from eth_typing import Address, BlockNumber, ChecksumAddress, HexStr
from eth_utils.toolz import merge
from hexbytes import HexBytes
from web3._utils.async_transactions import async_get_required_transaction, async_replace_transaction
from web3._utils.blocks import select_method_for_block_identifier
from web3._utils.fee_utils import async_fee_history_priority_fee
from web3._utils.filters import AsyncFilter, select_filter_method
from web3._utils.rpc_abi import RPC
from web3._utils.transactions import assert_valid_transaction_params, extract_valid_transaction_params
from web3.contract import AsyncContract, AsyncContractCaller
from web3.eth.base_eth import BaseEth
from web3.exceptions import MethodUnavailable, OffchainLookup, TimeExhausted, TooManyRequests, TransactionIndexingInProgress, TransactionNotFound
from web3.method import Method, default_root_munger
from web3.providers import PersistentConnectionProvider
from web3.types import ENS, BlockData, BlockIdentifier, BlockParams, CallOverride, CreateAccessListResponse, FeeHistory, FilterParams, LogReceipt, LogsSubscriptionArg, Nonce, SignedTx, SubscriptionType, SyncStatus, TxData, TxParams, TxReceipt, Wei, _Hash32
from web3.utils import async_handle_offchain_lookup
if TYPE_CHECKING:
    from web3 import AsyncWeb3

class AsyncEth(BaseEth):
    w3: 'AsyncWeb3'
    is_async = True
    _default_contract_factory: Type[Union[AsyncContract, AsyncContractCaller]] = AsyncContract
    _accounts: Method[Callable[[], Awaitable[Tuple[ChecksumAddress]]]] = Method(RPC.eth_accounts, is_property=True)
    _hashrate: Method[Callable[[], Awaitable[int]]] = Method(RPC.eth_hashrate, is_property=True)
    get_block_number: Method[Callable[[], Awaitable[BlockNumber]]] = Method(RPC.eth_blockNumber, is_property=True)
    _chain_id: Method[Callable[[], Awaitable[int]]] = Method(RPC.eth_chainId, is_property=True)
    _coinbase: Method[Callable[[], Awaitable[ChecksumAddress]]] = Method(RPC.eth_coinbase, is_property=True)
    _gas_price: Method[Callable[[], Awaitable[Wei]]] = Method(RPC.eth_gasPrice, is_property=True)
    _max_priority_fee: Method[Callable[[], Awaitable[Wei]]] = Method(RPC.eth_maxPriorityFeePerGas, is_property=True)

    @property
    async def max_priority_fee(self) -> Wei:
        """
        Try to use eth_maxPriorityFeePerGas but, since this is not part
        of the spec and is only supported by some clients, fall back to
        an eth_feeHistory calculation with min and max caps.
        """
        pass
    _mining: Method[Callable[[], Awaitable[bool]]] = Method(RPC.eth_mining, is_property=True)
    _syncing: Method[Callable[[], Awaitable[Union[SyncStatus, bool]]]] = Method(RPC.eth_syncing, is_property=True)
    _fee_history: Method[Callable[[int, Union[BlockParams, BlockNumber], Optional[List[float]]], Awaitable[FeeHistory]]] = Method(RPC.eth_feeHistory, mungers=[default_root_munger])
    _call: Method[Callable[[TxParams, Optional[BlockIdentifier], Optional[CallOverride]], Awaitable[HexBytes]]] = Method(RPC.eth_call, mungers=[BaseEth.call_munger])
    _create_access_list: Method[Callable[[TxParams, Optional[BlockIdentifier]], Awaitable[CreateAccessListResponse]]] = Method(RPC.eth_createAccessList, mungers=[BaseEth.create_access_list_munger])
    _estimate_gas: Method[Callable[[TxParams, Optional[BlockIdentifier], Optional[CallOverride]], Awaitable[int]]] = Method(RPC.eth_estimateGas, mungers=[BaseEth.estimate_gas_munger])
    _get_transaction: Method[Callable[[_Hash32], Awaitable[TxData]]] = Method(RPC.eth_getTransactionByHash, mungers=[default_root_munger])
    _get_raw_transaction: Method[Callable[[_Hash32], Awaitable[HexBytes]]] = Method(RPC.eth_getRawTransactionByHash, mungers=[default_root_munger])
    _get_transaction_by_block: Method[Callable[[BlockIdentifier, int], Awaitable[TxData]]] = Method(method_choice_depends_on_args=select_method_for_block_identifier(if_predefined=RPC.eth_getTransactionByBlockNumberAndIndex, if_hash=RPC.eth_getTransactionByBlockHashAndIndex, if_number=RPC.eth_getTransactionByBlockNumberAndIndex), mungers=[default_root_munger])
    _get_raw_transaction_by_block: Method[Callable[[BlockIdentifier, int], Awaitable[HexBytes]]] = Method(method_choice_depends_on_args=select_method_for_block_identifier(if_predefined=RPC.eth_getRawTransactionByBlockNumberAndIndex, if_hash=RPC.eth_getRawTransactionByBlockHashAndIndex, if_number=RPC.eth_getRawTransactionByBlockNumberAndIndex), mungers=[default_root_munger])
    get_block_transaction_count: Method[Callable[[BlockIdentifier], Awaitable[int]]] = Method(method_choice_depends_on_args=select_method_for_block_identifier(if_predefined=RPC.eth_getBlockTransactionCountByNumber, if_hash=RPC.eth_getBlockTransactionCountByHash, if_number=RPC.eth_getBlockTransactionCountByNumber), mungers=[default_root_munger])
    _send_transaction: Method[Callable[[TxParams], Awaitable[HexBytes]]] = Method(RPC.eth_sendTransaction, mungers=[BaseEth.send_transaction_munger])
    _send_raw_transaction: Method[Callable[[Union[HexStr, bytes]], Awaitable[HexBytes]]] = Method(RPC.eth_sendRawTransaction, mungers=[default_root_munger])
    _get_block: Method[Callable[[BlockIdentifier, bool], Awaitable[BlockData]]] = Method(method_choice_depends_on_args=select_method_for_block_identifier(if_predefined=RPC.eth_getBlockByNumber, if_hash=RPC.eth_getBlockByHash, if_number=RPC.eth_getBlockByNumber), mungers=[BaseEth.get_block_munger])
    _get_balance: Method[Callable[[Union[Address, ChecksumAddress, ENS], Optional[BlockIdentifier]], Awaitable[Wei]]] = Method(RPC.eth_getBalance, mungers=[BaseEth.block_id_munger])
    _get_code: Method[Callable[[Union[Address, ChecksumAddress, ENS], Optional[BlockIdentifier]], Awaitable[HexBytes]]] = Method(RPC.eth_getCode, mungers=[BaseEth.block_id_munger])
    _get_logs: Method[Callable[[FilterParams], Awaitable[List[LogReceipt]]]] = Method(RPC.eth_getLogs, mungers=[default_root_munger])
    _get_transaction_count: Method[Callable[[Union[Address, ChecksumAddress, ENS], Optional[BlockIdentifier]], Awaitable[Nonce]]] = Method(RPC.eth_getTransactionCount, mungers=[BaseEth.block_id_munger])
    _transaction_receipt: Method[Callable[[_Hash32], Awaitable[TxReceipt]]] = Method(RPC.eth_getTransactionReceipt, mungers=[default_root_munger])
    _get_storage_at: Method[Callable[[Union[Address, ChecksumAddress, ENS], int, Optional[BlockIdentifier]], Awaitable[HexBytes]]] = Method(RPC.eth_getStorageAt, mungers=[BaseEth.get_storage_at_munger])
    _sign: Method[Callable[..., Awaitable[HexStr]]] = Method(RPC.eth_sign, mungers=[BaseEth.sign_munger])
    _sign_transaction: Method[Callable[[TxParams], Awaitable[SignedTx]]] = Method(RPC.eth_signTransaction, mungers=[default_root_munger])
    _sign_typed_data: Method[Callable[[Union[Address, ChecksumAddress, ENS], Dict[str, Any]], Awaitable[HexStr]]] = Method(RPC.eth_signTypedData, mungers=[default_root_munger])
    _get_uncle_count: Method[Callable[[BlockIdentifier], Awaitable[int]]] = Method(method_choice_depends_on_args=select_method_for_block_identifier(if_predefined=RPC.eth_getUncleCountByBlockNumber, if_hash=RPC.eth_getUncleCountByBlockHash, if_number=RPC.eth_getUncleCountByBlockNumber), mungers=[default_root_munger])
    filter: Method[Callable[[Optional[Union[str, FilterParams, HexStr]]], Awaitable[AsyncFilter]]] = Method(method_choice_depends_on_args=select_filter_method(if_new_block_filter=RPC.eth_newBlockFilter, if_new_pending_transaction_filter=RPC.eth_newPendingTransactionFilter, if_new_filter=RPC.eth_newFilter), mungers=[BaseEth.filter_munger])
    _get_filter_changes: Method[Callable[[HexStr], Awaitable[List[LogReceipt]]]] = Method(RPC.eth_getFilterChanges, mungers=[default_root_munger])
    _get_filter_logs: Method[Callable[[HexStr], Awaitable[List[LogReceipt]]]] = Method(RPC.eth_getFilterLogs, mungers=[default_root_munger])
    _uninstall_filter: Method[Callable[[HexStr], Awaitable[bool]]] = Method(RPC.eth_uninstallFilter, mungers=[default_root_munger])
    _subscribe: Method[Callable[[SubscriptionType], Awaitable[HexStr]]] = Method(RPC.eth_subscribe, mungers=[default_root_munger])
    _subscribe_with_args: Method[Callable[[SubscriptionType, Optional[Union[LogsSubscriptionArg, bool]]], Awaitable[HexStr]]] = Method(RPC.eth_subscribe, mungers=[default_root_munger])
    _unsubscribe: Method[Callable[[HexStr], Awaitable[bool]]] = Method(RPC.eth_unsubscribe, mungers=[default_root_munger])