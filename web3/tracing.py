from typing import Callable, List, Optional, Tuple, Union
from eth_typing import HexStr
from eth_utils import is_checksum_address
from eth_utils.toolz import assoc
from web3._utils.rpc_abi import RPC
from web3.method import Method, default_root_munger
from web3.module import Module
from web3.types import BlockIdentifier, BlockTrace, FilterTrace, TraceFilterParams, TraceMode, TxParams, _Hash32

class Tracing(Module):
    """
    https://openethereum.github.io/JSONRPC-trace-module
    """
    _default_block: BlockIdentifier = 'latest'
    trace_replay_transaction: Method[Callable[..., BlockTrace]] = Method(RPC.trace_replayTransaction, mungers=[trace_replay_transaction_munger])
    trace_replay_block_transactions: Method[Callable[..., List[BlockTrace]]] = Method(RPC.trace_replayBlockTransactions, mungers=[trace_replay_transaction_munger])
    trace_block: Method[Callable[[BlockIdentifier], List[BlockTrace]]] = Method(RPC.trace_block, mungers=[default_root_munger])
    trace_filter: Method[Callable[[TraceFilterParams], List[FilterTrace]]] = Method(RPC.trace_filter, mungers=[default_root_munger])
    trace_transaction: Method[Callable[[_Hash32], List[FilterTrace]]] = Method(RPC.trace_transaction, mungers=[default_root_munger])
    trace_call: Method[Callable[..., BlockTrace]] = Method(RPC.trace_call, mungers=[trace_call_munger])
    trace_raw_transaction: Method[Callable[..., BlockTrace]] = Method(RPC.trace_rawTransaction, mungers=[trace_transactions_munger])