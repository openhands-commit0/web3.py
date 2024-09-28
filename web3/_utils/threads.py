"""
A minimal implementation of the various gevent APIs used within this codebase.
"""
import asyncio
import threading
import time
from types import TracebackType
from typing import Any, Callable, Generic, Type
from web3._utils.compat import Literal
from web3.types import TReturn

class Timeout(Exception):
    """
    A limited subset of the `gevent.Timeout` context manager.
    """
    seconds = None
    exception = None
    begun_at = None
    is_running = None

    def __init__(self, seconds: float=None, exception: Type[BaseException]=None, *args: Any, **kwargs: Any) -> None:
        self.seconds = seconds
        self.exception = exception

    def __enter__(self) -> 'Timeout':
        self.start()
        return self

    def __exit__(self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> Literal[False]:
        return False

    def __str__(self) -> str:
        if self.seconds is None:
            return ''
        return f'{self.seconds} seconds'

class ThreadWithReturn(threading.Thread, Generic[TReturn]):

    def __init__(self, target: Callable[..., TReturn]=None, args: Any=None, kwargs: Any=None) -> None:
        super().__init__(target=target, args=args or tuple(), kwargs=kwargs or {})
        self.target = target
        self.args = args
        self.kwargs = kwargs

class TimerClass(threading.Thread):

    def __init__(self, interval: int, callback: Callable[..., Any], *args: Any) -> None:
        threading.Thread.__init__(self)
        self.callback = callback
        self.terminate_event = threading.Event()
        self.interval = interval
        self.args = args