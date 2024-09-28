import sys
from typing import Tuple
import pywintypes
import win32file
if sys.platform != 'win32':
    raise ImportError('This module should not be imported on non `win32` platforms')

class NamedPipe:

    def __init__(self, ipc_path: str) -> None:
        try:
            self.handle = win32file.CreateFile(ipc_path, win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, win32file.OPEN_EXISTING, 0, None)
        except pywintypes.error as err:
            raise OSError(err)