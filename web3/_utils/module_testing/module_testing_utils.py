import pytest
import time
from typing import TYPE_CHECKING, Any, Collection, Dict, Generator, Optional, Sequence, Union
from aiohttp import ClientTimeout
from eth_typing import ChecksumAddress, HexStr
from eth_utils import is_same_address
from flaky import flaky
from hexbytes import HexBytes
from web3._utils.compat import Literal
from web3._utils.request import async_cache_and_return_session, asyncio, cache_and_return_session
from web3.types import BlockData, LogReceipt
if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from aiohttp import ClientResponse
    from requests import Response
    from web3 import Web3
    from web3._utils.compat import Self
'\nflaky_geth_dev_mining decorator for tests requiring a pending block\nfor the duration of the test. This behavior can be flaky\ndue to timing of the test running as a block is mined.\n'
flaky_geth_dev_mining = flaky(max_runs=3)

class WebsocketMessageStreamMock:
    closed: bool = False

    def __init__(self, messages: Optional[Collection[bytes]]=None, raise_exception: Optional[Exception]=None) -> None:
        self.queue = asyncio.Queue()
        for msg in messages or []:
            self.queue.put_nowait(msg)
        self.raise_exception = raise_exception

    def __await__(self) -> Generator[Any, Any, 'Self']:

        async def __async_init__() -> 'Self':
            return self
        return __async_init__().__await__()

    def __aiter__(self) -> 'Self':
        return self

    async def __anext__(self) -> bytes:
        return await self.recv()