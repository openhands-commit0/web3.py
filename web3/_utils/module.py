import inspect
from io import UnsupportedOperation
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from web3.exceptions import Web3ValidationError
from web3.module import Module
if TYPE_CHECKING:
    from web3.main import BaseWeb3