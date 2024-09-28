import functools
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, Set, cast
from eth_utils import is_list_like
import lru
from web3._utils.caching import generate_cache_key
from web3._utils.compat import Literal, TypedDict
from web3.types import BlockData, BlockNumber, Middleware, RPCEndpoint, RPCResponse
from web3.utils.caching import SimpleCache
if TYPE_CHECKING:
    from web3 import Web3
SIMPLE_CACHE_RPC_WHITELIST = cast(Set[RPCEndpoint], ('web3_clientVersion', 'net_version', 'eth_getBlockTransactionCountByHash', 'eth_getUncleCountByBlockHash', 'eth_getBlockByHash', 'eth_getTransactionByHash', 'eth_getTransactionByBlockHashAndIndex', 'eth_getRawTransactionByHash', 'eth_getUncleByBlockHashAndIndex', 'eth_chainId'))

def construct_simple_cache_middleware(cache: SimpleCache=None, rpc_whitelist: Collection[RPCEndpoint]=None, should_cache_fn: Callable[[RPCEndpoint, Any, RPCResponse], bool]=_should_cache_response) -> Middleware:
    """
    Constructs a middleware which caches responses based on the request
    ``method`` and ``params``

    :param cache: A ``SimpleCache`` class.
    :param rpc_whitelist: A set of RPC methods which may have their responses cached.
    :param should_cache_fn: A callable which accepts ``method`` ``params`` and
        ``response`` and returns a boolean as to whether the response should be
        cached.
    """
    pass
_simple_cache_middleware = construct_simple_cache_middleware()
TIME_BASED_CACHE_RPC_WHITELIST = cast(Set[RPCEndpoint], {'eth_coinbase', 'eth_accounts'})

def construct_time_based_cache_middleware(cache_class: Callable[..., Dict[Any, Any]], cache_expire_seconds: int=15, rpc_whitelist: Collection[RPCEndpoint]=TIME_BASED_CACHE_RPC_WHITELIST, should_cache_fn: Callable[[RPCEndpoint, Any, RPCResponse], bool]=_should_cache_response) -> Middleware:
    """
    Constructs a middleware which caches responses based on the request
    ``method`` and ``params`` for a maximum amount of time as specified

    :param cache_class: Any dictionary-like object
    :param cache_expire_seconds: The number of seconds an item may be cached
        before it should expire.
    :param rpc_whitelist: A set of RPC methods which may have their responses cached.
    :param should_cache_fn: A callable which accepts ``method`` ``params`` and
        ``response`` and returns a boolean as to whether the response should be
        cached.
    """
    pass
_time_based_cache_middleware = construct_time_based_cache_middleware(cache_class=functools.partial(lru.LRU, 256))
BLOCK_NUMBER_RPC_WHITELIST = cast(Set[RPCEndpoint], {'eth_gasPrice', 'eth_blockNumber', 'eth_getBalance', 'eth_getStorageAt', 'eth_getTransactionCount', 'eth_getBlockTransactionCountByNumber', 'eth_getUncleCountByBlockNumber', 'eth_getCode', 'eth_call', 'eth_createAccessList', 'eth_estimateGas', 'eth_getBlockByNumber', 'eth_getTransactionByBlockNumberAndIndex', 'eth_getTransactionReceipt', 'eth_getUncleByBlockNumberAndIndex', 'eth_getLogs'})
AVG_BLOCK_TIME_KEY: Literal['avg_block_time'] = 'avg_block_time'
AVG_BLOCK_SAMPLE_SIZE_KEY: Literal['avg_block_sample_size'] = 'avg_block_sample_size'
AVG_BLOCK_TIME_UPDATED_AT_KEY: Literal['avg_block_time_updated_at'] = 'avg_block_time_updated_at'
BlockInfoCache = TypedDict('BlockInfoCache', {'avg_block_time': float, 'avg_block_sample_size': int, 'avg_block_time_updated_at': float, 'latest_block': BlockData}, total=False)

def construct_latest_block_based_cache_middleware(cache_class: Callable[..., Dict[Any, Any]], rpc_whitelist: Collection[RPCEndpoint]=BLOCK_NUMBER_RPC_WHITELIST, average_block_time_sample_size: int=240, default_average_block_time: int=15, should_cache_fn: Callable[[RPCEndpoint, Any, RPCResponse], bool]=_should_cache_response) -> Middleware:
    """
    Constructs a middleware which caches responses based on the request
    ``method``, ``params``, and the current latest block hash.

    :param cache_class: Any dictionary-like object
    :param rpc_whitelist: A set of RPC methods which may have their responses cached.
    :param average_block_time_sample_size: number of blocks to look back when computing
        average block time
    :param default_average_block_time: estimated number of seconds per block
    :param should_cache_fn: A callable which accepts ``method`` ``params`` and
        ``response`` and returns a boolean as to whether the response should be
        cached.

    .. note::
        This middleware avoids re-fetching the current latest block for each
        request by tracking the current average block time and only requesting
        a new block when the last seen latest block is older than the average
        block time.
    """
    pass
_latest_block_based_cache_middleware = construct_latest_block_based_cache_middleware(cache_class=functools.partial(lru.LRU, 256), rpc_whitelist=BLOCK_NUMBER_RPC_WHITELIST)