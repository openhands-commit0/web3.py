from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

class SimpleCache:

    def __init__(self, size: int=100):
        self._size = size
        self._data: OrderedDict[str, Any] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)