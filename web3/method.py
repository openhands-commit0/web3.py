import functools
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypeVar, Union
import warnings
from eth_utils.curried import to_tuple
from eth_utils.toolz import pipe
from web3._utils.method_formatters import get_error_formatters, get_null_result_formatters, get_request_formatters, get_result_formatters
from web3._utils.rpc_abi import RPC
from web3.exceptions import Web3ValidationError
from web3.types import RPCEndpoint, TReturn
if TYPE_CHECKING:
    from web3 import Web3
    from web3.module import Module
Munger = Callable[..., Any]
TFunc = TypeVar('TFunc', bound=Callable[..., Any])

class Method(Generic[TFunc]):
    """Method object for web3 module methods

    Calls to the Method go through these steps:

    1. input munging - includes normalization, parameter checking, early parameter
    formatting. Any processing on the input parameters that need to happen before
    json_rpc method string selection occurs.

            A note about mungers: The first (root) munger should reflect the desired
        api function arguments. In other words, if the api function wants to
        behave as: `get_balance(account, block_identifier=None)`, the root munger
        should accept these same arguments, with the addition of the module as
        the first argument e.g.:

        ```
        def get_balance_root_munger(module, account, block_identifier=None):
            if block_identifier is None:
                block_identifier = DEFAULT_BLOCK
            return module, [account, block_identifier]
        ```

        all mungers should return an argument list.

        if no munger is provided, a default munger expecting no method arguments
        will be used.

    2. method selection - The json_rpc_method argument can be method string or a
    function that returns a method string. If a callable is provided the processed
    method inputs are passed to the method selection function, and the returned
    method string is used.

    3. request and response formatters are set - formatters are retrieved
    using the json rpc method string.

    4. After the parameter processing from steps 1-3 the request is made using
    the calling function returned by the module attribute ``retrieve_caller_fn``
    and the response formatters are applied to the output.
    """

    def __init__(self, json_rpc_method: Optional[RPCEndpoint]=None, mungers: Optional[Sequence[Munger]]=None, request_formatters: Optional[Callable[..., TReturn]]=None, result_formatters: Optional[Callable[..., TReturn]]=None, null_result_formatters: Optional[Callable[..., TReturn]]=None, method_choice_depends_on_args: Optional[Callable[..., RPCEndpoint]]=None, is_property: bool=False):
        self.json_rpc_method = json_rpc_method
        self.mungers = _set_mungers(mungers, is_property)
        self.request_formatters = request_formatters or get_request_formatters
        self.result_formatters = result_formatters or get_result_formatters
        self.null_result_formatters = null_result_formatters or get_null_result_formatters
        self.method_choice_depends_on_args = method_choice_depends_on_args
        self.is_property = is_property

    def __get__(self, obj: Optional['Module']=None, obj_type: Optional[Type['Module']]=None) -> TFunc:
        if obj is None:
            raise TypeError('Direct calls to methods are not supported. Methods must be called from an module instance, usually attached to a web3 instance.')
        return obj.retrieve_caller_fn(self)

    @property
    def method_selector_fn(self) -> Callable[..., Union[RPCEndpoint, Callable[..., RPCEndpoint]]]:
        """Gets the method selector from the config."""
        pass

class DeprecatedMethod:

    def __init__(self, method: Method[Callable[..., Any]], old_name: Optional[str]=None, new_name: Optional[str]=None, msg: Optional[str]=None) -> None:
        self.method = method
        self.old_name = old_name
        self.new_name = new_name
        self.msg = msg

    def __get__(self, obj: Optional['Module']=None, obj_type: Optional[Type['Module']]=None) -> Any:
        if self.old_name is not None and self.new_name is not None:
            if self.msg is not None:
                raise ValueError('Cannot specify `old_name` and `new_name` along with `msg`')
            message = f'{self.old_name} is deprecated in favor of {self.new_name}'
        elif self.msg is not None:
            message = self.msg
        else:
            raise ValueError('Must provide either `old_name` and `new_name` or `msg`')
        warnings.warn(message, category=DeprecationWarning)
        return self.method.__get__(obj, obj_type)