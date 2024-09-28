import collections
import hashlib
from typing import TYPE_CHECKING, Any, Callable, List, Tuple
from eth_utils import is_boolean, is_bytes, is_dict, is_list_like, is_null, is_number, is_text, to_bytes
if TYPE_CHECKING:
    from web3.types import RPCEndpoint

def generate_cache_key(value: Any) -> str:
    """
    Generates a cache key for the *args and **kwargs
    """
    pass

class RequestInformation:

    def __init__(self, method: 'RPCEndpoint', params: Any, response_formatters: Tuple[Callable[..., Any], ...], subscription_id: str=None):
        self.method = method
        self.params = params
        self.response_formatters = response_formatters
        self.subscription_id = subscription_id
        self.middleware_response_processors: List[Callable[..., Any]] = []