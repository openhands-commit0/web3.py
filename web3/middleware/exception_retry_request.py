import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable, Collection, List, Optional, Type
import aiohttp
from requests.exceptions import ConnectionError, HTTPError, Timeout, TooManyRedirects
from web3.types import AsyncMiddlewareCoroutine, RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3
DEFAULT_ALLOWLIST = ['admin', 'miner', 'net', 'txpool', 'testing', 'evm', 'eth_protocolVersion', 'eth_syncing', 'eth_coinbase', 'eth_mining', 'eth_hashrate', 'eth_chainId', 'eth_gasPrice', 'eth_accounts', 'eth_blockNumber', 'eth_getBalance', 'eth_getStorageAt', 'eth_getProof', 'eth_getCode', 'eth_getBlockByNumber', 'eth_getBlockByHash', 'eth_getBlockTransactionCountByNumber', 'eth_getBlockTransactionCountByHash', 'eth_getUncleCountByBlockNumber', 'eth_getUncleCountByBlockHash', 'eth_getTransactionByHash', 'eth_getTransactionByBlockHashAndIndex', 'eth_getTransactionByBlockNumberAndIndex', 'eth_getTransactionReceipt', 'eth_getTransactionCount', 'eth_getRawTransactionByHash', 'eth_call', 'eth_createAccessList', 'eth_estimateGas', 'eth_maxPriorityFeePerGas', 'eth_newBlockFilter', 'eth_newPendingTransactionFilter', 'eth_newFilter', 'eth_getFilterChanges', 'eth_getFilterLogs', 'eth_getLogs', 'eth_uninstallFilter', 'eth_getCompilers', 'eth_getWork', 'eth_sign', 'eth_signTypedData', 'eth_sendRawTransaction', 'personal_importRawKey', 'personal_newAccount', 'personal_listAccounts', 'personal_listWallets', 'personal_lockAccount', 'personal_unlockAccount', 'personal_ecRecover', 'personal_sign', 'personal_signTypedData']

def exception_retry_middleware(make_request: Callable[[RPCEndpoint, Any], RPCResponse], _w3: 'Web3', errors: Collection[Type[BaseException]], retries: int=5, backoff_factor: float=0.3, allow_list: Optional[List[str]]=None) -> Callable[[RPCEndpoint, Any], RPCResponse]:
    """
    Creates middleware that retries failed HTTP requests. Is a default
    middleware for HTTPProvider.
    """
    pass

async def async_exception_retry_middleware(make_request: Callable[[RPCEndpoint, Any], Any], _async_w3: 'AsyncWeb3', errors: Collection[Type[BaseException]], retries: int=5, backoff_factor: float=0.3, allow_list: Optional[List[str]]=None) -> AsyncMiddlewareCoroutine:
    """
    Creates middleware that retries failed HTTP requests.
    Is a default middleware for AsyncHTTPProvider.
    """
    pass