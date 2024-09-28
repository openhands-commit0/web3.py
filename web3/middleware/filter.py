import itertools
import os
from typing import TYPE_CHECKING, Any, AsyncIterable, AsyncIterator, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union, cast
from eth_typing import Address, BlockNumber, ChecksumAddress, Hash32, HexStr
from eth_utils import apply_key_map, is_hex, is_string, to_hex, to_int, to_list
from eth_utils.toolz import concat, valfilter
from web3._utils.formatters import hex_to_integer
from web3._utils.rpc_abi import RPC
from web3.types import Coroutine, FilterParams, LatestBlockParam, LogReceipt, RPCEndpoint, RPCResponse, _Hash32
if TYPE_CHECKING:
    from web3 import Web3
if 'WEB3_MAX_BLOCK_REQUEST' in os.environ:
    MAX_BLOCK_REQUEST = to_int(text=os.environ['WEB3_MAX_BLOCK_REQUEST'])
else:
    MAX_BLOCK_REQUEST = 50

def segment_count(start: int, stop: int, step: int=5) -> Iterable[Tuple[int, int]]:
    """Creates a segment counting generator

    The generator returns tuple pairs of integers
    that correspond to segments in the provided range.

    :param start: The initial value of the counting range
    :param stop: The last value in the
    counting range
    :param step: Optional, the segment length. Default is 5.
    :return: returns a generator object


    Example:

    >>> segment_counter = segment_count(start=0, stop=10, step=3)
    >>> next(segment_counter)
    (0, 3)
    >>> next(segment_counter)
    (3, 6)
    >>> next(segment_counter)
    (6, 9)
    >>> next(segment_counter) #  Remainder is also returned
    (9, 10)
    """
    pass

def block_ranges(start_block: BlockNumber, last_block: Optional[BlockNumber], step: int=5) -> Iterable[Tuple[BlockNumber, BlockNumber]]:
    """Returns 2-tuple ranges describing ranges of block from start_block to last_block

    Ranges do not overlap to facilitate use as ``toBlock``, ``fromBlock``
    json-rpc arguments, which are both inclusive.
    """
    pass

def iter_latest_block(w3: 'Web3', to_block: Optional[Union[BlockNumber, LatestBlockParam]]=None) -> Iterable[BlockNumber]:
    """Returns a generator that dispenses the latest block, if
    any new blocks have been mined since last iteration.

    If there are no new blocks or the latest block is greater than
    the ``to_block`` None is returned.

    >>> new_blocks = iter_latest_block(w3, 0, 10)
    >>> next(new_blocks)  # Latest block = 0
    0
    >>> next(new_blocks)  # No new blocks
    >>> next(new_blocks)  # Latest block = 1
    1
    >>> next(new_blocks)  # Latest block = 10
    10
    >>> next(new_blocks)  # latest block > to block
    """
    pass

def iter_latest_block_ranges(w3: 'Web3', from_block: BlockNumber, to_block: Optional[Union[BlockNumber, LatestBlockParam]]=None) -> Iterable[Tuple[Optional[BlockNumber], Optional[BlockNumber]]]:
    """Returns an iterator unloading ranges of available blocks

    starting from `fromBlock` to the latest mined block,
    until reaching toBlock. e.g.:


    >>> blocks_to_filter = iter_latest_block_ranges(w3, 0, 50)
    >>> next(blocks_to_filter)  # latest block number = 11
    (0, 11)
    >>> next(blocks_to_filter)  # latest block number = 45
    (12, 45)
    >>> next(blocks_to_filter)  # latest block number = 50
    (46, 50)
    """
    pass

def get_logs_multipart(w3: 'Web3', start_block: BlockNumber, stop_block: BlockNumber, address: Union[Address, ChecksumAddress, List[Union[Address, ChecksumAddress]]], topics: List[Optional[Union[_Hash32, List[_Hash32]]]], max_blocks: int) -> Iterable[List[LogReceipt]]:
    """Used to break up requests to ``eth_getLogs``

    The getLog request is partitioned into multiple calls of the max number of blocks
    ``max_blocks``.
    """
    pass

class RequestLogs:
    _from_block: BlockNumber

    def __init__(self, w3: 'Web3', from_block: Optional[Union[BlockNumber, LatestBlockParam]]=None, to_block: Optional[Union[BlockNumber, LatestBlockParam]]=None, address: Optional[Union[Address, ChecksumAddress, List[Union[Address, ChecksumAddress]]]]=None, topics: Optional[List[Optional[Union[_Hash32, List[_Hash32]]]]]=None) -> None:
        self.address = address
        self.topics = topics
        self.w3 = w3
        if from_block is None or from_block == 'latest':
            self._from_block = BlockNumber(w3.eth.block_number + 1)
        elif is_string(from_block) and is_hex(from_block):
            self._from_block = BlockNumber(hex_to_integer(from_block))
        else:
            self._from_block = from_block
        self._to_block = to_block
        self.filter_changes = self._get_filter_changes()
FILTER_PARAMS_KEY_MAP = {'toBlock': 'to_block', 'fromBlock': 'from_block'}
NEW_FILTER_METHODS = {'eth_newBlockFilter', 'eth_newFilter'}
FILTER_CHANGES_METHODS = {'eth_getFilterChanges', 'eth_getFilterLogs'}

class RequestBlocks:

    def __init__(self, w3: 'Web3') -> None:
        self.w3 = w3
        self.start_block = BlockNumber(w3.eth.block_number + 1)

async def async_iter_latest_block(w3: 'Web3', to_block: Optional[Union[BlockNumber, LatestBlockParam]]=None) -> AsyncIterable[BlockNumber]:
    """Returns a generator that dispenses the latest block, if
    any new blocks have been mined since last iteration.

    If there are no new blocks or the latest block is greater than
    the ``to_block`` None is returned.

    >>> new_blocks = iter_latest_block(w3, 0, 10)
    >>> next(new_blocks)  # Latest block = 0
    0
    >>> next(new_blocks)  # No new blocks
    >>> next(new_blocks)  # Latest block = 1
    1
    >>> next(new_blocks)  # Latest block = 10
    10
    >>> next(new_blocks)  # latest block > to block
    """
    pass

async def async_iter_latest_block_ranges(w3: 'Web3', from_block: BlockNumber, to_block: Optional[Union[BlockNumber, LatestBlockParam]]=None) -> AsyncIterable[Tuple[Optional[BlockNumber], Optional[BlockNumber]]]:
    """Returns an iterator unloading ranges of available blocks

    starting from `from_block` to the latest mined block,
    until reaching to_block. e.g.:


    >>> blocks_to_filter = iter_latest_block_ranges(w3, 0, 50)
    >>> next(blocks_to_filter)  # latest block number = 11
    (0, 11)
    >>> next(blocks_to_filter)  # latest block number = 45
    (12, 45)
    >>> next(blocks_to_filter)  # latest block number = 50
    (46, 50)
    """
    pass

async def async_get_logs_multipart(w3: 'Web3', start_block: BlockNumber, stop_block: BlockNumber, address: Union[Address, ChecksumAddress, List[Union[Address, ChecksumAddress]]], topics: List[Optional[Union[_Hash32, List[_Hash32]]]], max_blocks: int) -> AsyncIterable[List[LogReceipt]]:
    """Used to break up requests to ``eth_getLogs``

    The getLog request is partitioned into multiple calls of the max number of blocks
    ``max_blocks``.
    """
    pass

class AsyncRequestLogs:
    _from_block: BlockNumber

    def __init__(self, w3: 'Web3', from_block: Optional[Union[BlockNumber, LatestBlockParam]]=None, to_block: Optional[Union[BlockNumber, LatestBlockParam]]=None, address: Optional[Union[Address, ChecksumAddress, List[Union[Address, ChecksumAddress]]]]=None, topics: Optional[List[Optional[Union[_Hash32, List[_Hash32]]]]]=None) -> None:
        self.address = address
        self.topics = topics
        self.w3 = w3
        self._from_block_arg = from_block
        self._to_block = to_block
        self.filter_changes = self._get_filter_changes()

    def __await__(self) -> Generator[Any, None, 'AsyncRequestLogs']:

        async def closure() -> 'AsyncRequestLogs':
            if self._from_block_arg is None or self._from_block_arg == 'latest':
                self.block_number = await self.w3.eth.block_number
                self._from_block = BlockNumber(self.block_number + 1)
            elif is_string(self._from_block_arg) and is_hex(self._from_block_arg):
                self._from_block = BlockNumber(hex_to_integer(cast(HexStr, self._from_block_arg)))
            else:
                self._from_block = self._from_block_arg
            return self
        return closure().__await__()

class AsyncRequestBlocks:

    def __init__(self, w3: 'Web3') -> None:
        self.w3 = w3

    def __await__(self) -> Generator[Any, None, 'AsyncRequestBlocks']:

        async def closure() -> 'AsyncRequestBlocks':
            self.block_number = await self.w3.eth.block_number
            self.start_block = BlockNumber(self.block_number + 1)
            return self
        return closure().__await__()