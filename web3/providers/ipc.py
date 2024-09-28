from json import JSONDecodeError
import logging
import os
from pathlib import Path
import socket
import sys
import threading
from types import TracebackType
from typing import Any, Optional, Type, Union
from web3._utils.threads import Timeout
from web3.types import RPCEndpoint, RPCResponse
from .base import JSONBaseProvider

class PersistantSocket:
    sock = None

    def __init__(self, ipc_path: str) -> None:
        self.ipc_path = ipc_path

    def __enter__(self) -> socket.socket:
        if not self.ipc_path:
            raise FileNotFoundError(f'cannot connect to IPC socket at path: {self.ipc_path!r}')
        if not self.sock:
            self.sock = self._open()
        return self.sock

    def __exit__(self, exc_type: Type[BaseException], exc_value: BaseException, traceback: TracebackType) -> None:
        if exc_value is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

class IPCProvider(JSONBaseProvider):
    logger = logging.getLogger('web3.providers.IPCProvider')
    _socket = None

    def __init__(self, ipc_path: Union[str, Path]=None, timeout: int=10, *args: Any, **kwargs: Any) -> None:
        if ipc_path is None:
            self.ipc_path = get_default_ipc_path()
        elif isinstance(ipc_path, str) or isinstance(ipc_path, Path):
            self.ipc_path = str(Path(ipc_path).expanduser().resolve())
        else:
            raise TypeError('ipc_path must be of type string or pathlib.Path')
        self.timeout = timeout
        self._lock = threading.Lock()
        self._socket = PersistantSocket(self.ipc_path)
        super().__init__()

    def __str__(self) -> str:
        return f'<{self.__class__.__name__} {self.ipc_path}>'