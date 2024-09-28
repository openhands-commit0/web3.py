from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
from eth_abi import abi
from eth_abi.exceptions import DecodingError
from eth_utils import is_bytes
from web3._utils.compat import Literal
from web3.middleware.attrdict import async_attrdict_middleware, attrdict_middleware
from web3.middleware.buffered_gas_estimate import async_buffered_gas_estimate_middleware
from web3.providers import BaseProvider
from web3.providers.async_base import AsyncBaseProvider
from web3.types import RPCEndpoint, RPCError, RPCResponse
from .middleware import async_default_transaction_fields_middleware, async_ethereum_tester_middleware, default_transaction_fields_middleware, ethereum_tester_middleware
if TYPE_CHECKING:
    from eth_tester import EthereumTester
    from eth_tester.backends.base import BaseChainBackend

class AsyncEthereumTesterProvider(AsyncBaseProvider):
    middlewares = (async_attrdict_middleware, async_buffered_gas_estimate_middleware, async_default_transaction_fields_middleware, async_ethereum_tester_middleware)

    def __init__(self) -> None:
        super().__init__()
        from eth_tester import EthereumTester
        from web3.providers.eth_tester.defaults import API_ENDPOINTS
        self.ethereum_tester = EthereumTester()
        self.api_endpoints = API_ENDPOINTS

class EthereumTesterProvider(BaseProvider):
    middlewares = (attrdict_middleware, default_transaction_fields_middleware, ethereum_tester_middleware)
    ethereum_tester = None
    api_endpoints: Optional[Dict[str, Dict[str, Callable[..., RPCResponse]]]] = None

    def __init__(self, ethereum_tester: Optional[Union['EthereumTester', 'BaseChainBackend']]=None, api_endpoints: Optional[Dict[str, Dict[str, Callable[..., RPCResponse]]]]=None) -> None:
        from eth_tester import EthereumTester
        from eth_tester.backends.base import BaseChainBackend
        if ethereum_tester is None:
            self.ethereum_tester = EthereumTester()
        elif isinstance(ethereum_tester, EthereumTester):
            self.ethereum_tester = ethereum_tester
        elif isinstance(ethereum_tester, BaseChainBackend):
            self.ethereum_tester = EthereumTester(ethereum_tester)
        else:
            raise TypeError(f'Expected ethereum_tester to be of type `eth_tester.EthereumTester` or a subclass of `eth_tester.backends.base.BaseChainBackend`, instead received {type(ethereum_tester)}. If you would like a custom eth-tester instance to test with, see the eth-tester documentation. https://github.com/ethereum/eth-tester.')
        if api_endpoints is None:
            from .defaults import API_ENDPOINTS
            self.api_endpoints = API_ENDPOINTS
        else:
            self.api_endpoints = api_endpoints