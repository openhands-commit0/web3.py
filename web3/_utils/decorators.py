import functools
import threading
from typing import Any, Callable, TypeVar, cast
import warnings
TFunc = TypeVar('TFunc', bound=Callable[..., Any])

def reject_recursive_repeats(to_wrap: Callable[..., Any]) -> Callable[..., Any]:
    """
    Prevent simple cycles by returning None when called recursively with same instance
    """
    pass

def deprecate_method(replacement_method: str=None, deprecation_msg: str=None) -> Callable[..., Any]:
    """
    Decorate a deprecated function with info on its replacement method OR a clarifying
    reason for the deprecation.

    @deprecate_method("to_bytes()")
    def to_ascii(arg):
        ...

    @deprecate_method(deprecation_msg=(
        "This method is no longer supported and will be removed in the next release."
    ))
    def some_method(arg):
        ...
    """
    pass