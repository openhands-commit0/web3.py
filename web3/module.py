from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Optional, TypeVar, Union, cast
from eth_abi.codec import ABICodec
from eth_utils.toolz import curry, pipe
from web3._utils.filters import AsyncLogFilter, LogFilter, _UseExistingFilter
from web3.method import Method
from web3.providers.persistent import PersistentConnectionProvider
from web3.types import RPCEndpoint, RPCResponse
if TYPE_CHECKING:
    from web3.main import AsyncWeb3, Web3
TReturn = TypeVar('TReturn')

class Module:
    is_async = False

    def __init__(self, w3: Union['AsyncWeb3', 'Web3']) -> None:
        if self.is_async:
            self.retrieve_caller_fn = retrieve_async_method_call_fn(w3, self)
        else:
            self.retrieve_caller_fn = retrieve_blocking_method_call_fn(w3, self)
        self.w3 = w3