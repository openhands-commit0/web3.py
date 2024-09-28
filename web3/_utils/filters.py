from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from eth_abi.codec import ABICodec
from eth_abi.grammar import parse as parse_type_string
from eth_typing import ChecksumAddress, HexStr, TypeStr
from eth_utils import is_hex, is_list_like, is_string, is_text
from eth_utils.curried import apply_formatter_if
from eth_utils.toolz import complement, curry
from hexbytes import HexBytes
from web3._utils.events import AsyncEventFilterBuilder, EventFilterBuilder, construct_event_data_set, construct_event_topic_set
from web3._utils.validation import validate_address
from web3.exceptions import Web3ValidationError
from web3.types import ABIEvent, BlockIdentifier, FilterParams, LogReceipt, RPCEndpoint
if TYPE_CHECKING:
    from web3.eth import AsyncEth
    from web3.eth import Eth

class BaseFilter:
    callbacks: List[Callable[..., Any]] = None
    stopped = False
    poll_interval = None
    filter_id = None

    def __init__(self, filter_id: HexStr) -> None:
        self.filter_id = filter_id
        self.callbacks = []
        super().__init__()

    def __str__(self) -> str:
        return f'Filter for {self.filter_id}'

    def format_entry(self, entry: LogReceipt) -> LogReceipt:
        """
        Hook for subclasses to change the format of the value that is passed
        into the callback functions.
        """
        pass

    def is_valid_entry(self, entry: LogReceipt) -> bool:
        """
        Hook for subclasses to implement additional filtering layers.
        """
        pass

class Filter(BaseFilter):

    def __init__(self, filter_id: HexStr, eth_module: 'Eth') -> None:
        self.eth_module = eth_module
        super(Filter, self).__init__(filter_id)

class AsyncFilter(BaseFilter):

    def __init__(self, filter_id: HexStr, eth_module: 'AsyncEth') -> None:
        self.eth_module = eth_module
        super(AsyncFilter, self).__init__(filter_id)

class BlockFilter(Filter):
    pass

class AsyncBlockFilter(AsyncFilter):
    pass

class TransactionFilter(Filter):
    pass

class AsyncTransactionFilter(AsyncFilter):
    pass

class LogFilter(Filter):
    data_filter_set = None
    data_filter_set_regex = None
    data_filter_set_function = None
    log_entry_formatter = None
    filter_params: FilterParams = None
    builder: EventFilterBuilder = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.log_entry_formatter = kwargs.pop('log_entry_formatter', self.log_entry_formatter)
        if 'data_filter_set' in kwargs:
            self.set_data_filters(kwargs.pop('data_filter_set'))
        super().__init__(*args, **kwargs)

    def set_data_filters(self, data_filter_set: Collection[Tuple[TypeStr, Any]]) -> None:
        """Sets the data filters (non indexed argument filters)

        Expects a set of tuples with the type and value, e.g.:
        (('uint256', [12345, 54321]), ('string', ('a-single-string',)))
        """
        pass

class AsyncLogFilter(AsyncFilter):
    data_filter_set = None
    data_filter_set_regex = None
    data_filter_set_function = None
    log_entry_formatter = None
    filter_params: FilterParams = None
    builder: AsyncEventFilterBuilder = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.log_entry_formatter = kwargs.pop('log_entry_formatter', self.log_entry_formatter)
        if 'data_filter_set' in kwargs:
            self.set_data_filters(kwargs.pop('data_filter_set'))
        super().__init__(*args, **kwargs)

    def set_data_filters(self, data_filter_set: Collection[Tuple[TypeStr, Any]]) -> None:
        """Sets the data filters (non indexed argument filters)

        Expects a set of tuples with the type and value, e.g.:
        (('uint256', [12345, 54321]), ('string', ('a-single-string',)))
        """
        pass
not_text = complement(is_text)
normalize_to_text = apply_formatter_if(not_text, decode_utf8_bytes)

def normalize_data_values(type_string: TypeStr, data_value: Any) -> Any:
    """Decodes utf-8 bytes to strings for abi string values.

    eth-abi v1 returns utf-8 bytes for string values.
    This can be removed once eth-abi v2 is required.
    """
    pass

@curry
def match_fn(codec: ABICodec, match_values_and_abi: Collection[Tuple[str, Any]], data: Any) -> bool:
    """Match function used for filtering non-indexed event arguments.

    Values provided through the match_values_and_abi parameter are
    compared to the abi decoded log data.
    """
    pass

class _UseExistingFilter(Exception):
    """
    Internal exception, raised when a filter_id is passed into w3.eth.filter()
    """

    def __init__(self, filter_id: Union[str, FilterParams, HexStr]) -> None:
        self.filter_id = filter_id