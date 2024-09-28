import decimal
import warnings
from types import TracebackType
from ens import AsyncENS, ENS
from eth_abi.codec import ABICodec
from eth_utils import add_0x_prefix, apply_to_return_value, from_wei, is_address, is_checksum_address, keccak as eth_utils_keccak, remove_0x_prefix, to_bytes, to_checksum_address, to_int, to_text, to_wei
from functools import wraps
from hexbytes import HexBytes
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Generator, List, Optional, Sequence, Type, Union, cast
from eth_typing import AnyAddress, ChecksumAddress, HexStr, Primitives
from eth_typing.abi import TypeStr
from eth_utils import combomethod
from web3._utils.abi import build_non_strict_registry, build_strict_registry, map_abi_data
from web3._utils.compat import Self
from web3._utils.empty import empty
from web3._utils.encoding import hex_encode_abi_type, to_hex, to_json
from web3._utils.rpc_abi import RPC
from web3._utils.module import attach_modules as _attach_modules
from web3._utils.normalizers import abi_ens_resolver
from web3.eth import AsyncEth, Eth
from web3.exceptions import Web3ValidationError
from web3.geth import AsyncGeth, AsyncGethAdmin, AsyncGethPersonal, AsyncGethTxPool, Geth, GethAdmin, GethMiner, GethPersonal, GethTxPool
from web3.manager import RequestManager as DefaultRequestManager
from web3.module import Module
from web3.net import AsyncNet, Net
from web3.providers import AsyncBaseProvider, BaseProvider
from web3.providers.eth_tester import EthereumTesterProvider
from web3.providers.ipc import IPCProvider
from web3.providers.async_rpc import AsyncHTTPProvider
from web3.providers.persistent import PersistentConnectionProvider
from web3.providers.rpc import HTTPProvider
from web3.providers.websocket import WebsocketProvider
from web3.providers.websocket.websocket_connection import WebsocketConnection
from web3.testing import Testing
from web3.tracing import Tracing
from web3.types import AsyncMiddlewareOnion, MiddlewareOnion, Wei
if TYPE_CHECKING:
    from web3.pm import PM
    from web3._utils.empty import Empty

class BaseWeb3:
    _strict_bytes_type_checking = True
    HTTPProvider = HTTPProvider
    IPCProvider = IPCProvider
    EthereumTesterProvider = EthereumTesterProvider
    WebsocketProvider = WebsocketProvider
    AsyncHTTPProvider = AsyncHTTPProvider
    RequestManager = DefaultRequestManager
    eth: Union[Eth, AsyncEth]
    net: Union[Net, AsyncNet]
    geth: Union[Geth, AsyncGeth]

    @combomethod
    def solidity_keccak(cls, abi_types: List[TypeStr], values: List[Any]) -> bytes:
        """
        Executes keccak256 exactly as Solidity does.
        Takes list of abi_types as inputs -- `[uint24, int8[], bool]`
        and list of corresponding values  -- `[20, [-1, 5, 0], True]`
        """
        pass

    def attach_modules(self, modules: Optional[Dict[str, Union[Type[Module], Sequence[Any]]]]) -> None:
        """
        Attach modules to the `Web3` instance.
        """
        pass

class Web3(BaseWeb3):
    eth: Eth
    net: Net
    geth: Geth

    def __init__(self, provider: Optional[BaseProvider]=None, middlewares: Optional[Sequence[Any]]=None, modules: Optional[Dict[str, Union[Type[Module], Sequence[Any]]]]=None, external_modules: Optional[Dict[str, Union[Type[Module], Sequence[Any]]]]=None, ens: Union[ENS, 'Empty']=empty) -> None:
        self.manager = self.RequestManager(self, provider, middlewares)
        self.codec = ABICodec(build_strict_registry())
        if modules is None:
            modules = get_default_modules()
        self.attach_modules(modules)
        if external_modules is not None:
            self.attach_modules(external_modules)
        self.ens = ens

class AsyncWeb3(BaseWeb3):
    eth: AsyncEth
    net: AsyncNet
    geth: AsyncGeth

    def __init__(self, provider: Optional[AsyncBaseProvider]=None, middlewares: Optional[Sequence[Any]]=None, modules: Optional[Dict[str, Union[Type[Module], Sequence[Any]]]]=None, external_modules: Optional[Dict[str, Union[Type[Module], Sequence[Any]]]]=None, ens: Union[AsyncENS, 'Empty']=empty, **kwargs: Any) -> None:
        self.manager = self.RequestManager(self, provider, middlewares)
        self.codec = ABICodec(build_strict_registry())
        if modules is None:
            modules = get_async_default_modules()
        self.attach_modules(modules)
        if external_modules is not None:
            self.attach_modules(external_modules)
        self.ens = ens

    @staticmethod
    def persistent_websocket(provider: PersistentConnectionProvider, middlewares: Optional[Sequence[Any]]=None, modules: Optional[Dict[str, Union[Type[Module], Sequence[Any]]]]=None, external_modules: Optional[Dict[str, Union[Type[Module], Sequence[Any]]]]=None, ens: Union[AsyncENS, 'Empty']=empty) -> '_PersistentConnectionWeb3':
        """
        Establish a persistent connection via websockets to a websocket provider using
        a ``PersistentConnectionProvider`` instance.
        """
        pass

class _PersistentConnectionWeb3(AsyncWeb3):
    provider: PersistentConnectionProvider

    def __init__(self, provider: PersistentConnectionProvider=None, middlewares: Optional[Sequence[Any]]=None, modules: Optional[Dict[str, Union[Type[Module], Sequence[Any]]]]=None, external_modules: Optional[Dict[str, Union[Type[Module], Sequence[Any]]]]=None, ens: Union[AsyncENS, 'Empty']=empty) -> None:
        if not isinstance(provider, PersistentConnectionProvider):
            raise Web3ValidationError('Provider must inherit from PersistentConnectionProvider class.')
        AsyncWeb3.__init__(self, provider, middlewares, modules, external_modules, ens)
        self.ws = WebsocketConnection(self)

    def __await__(self) -> Generator[Any, None, Self]:

        async def __async_init__() -> Self:
            if self.provider._ws is None:
                await self.provider.connect()
            return self
        return __async_init__().__await__()

    async def __aenter__(self) -> Self:
        await self.provider.connect()
        return self

    async def __aexit__(self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        await self.provider.disconnect()

    async def __aiter__(self) -> AsyncIterator[Self]:
        provider = self.provider
        while True:
            await provider.connect()
            yield self
            provider.logger.error('Connection interrupted, attempting to reconnect...')
            await provider.disconnect()