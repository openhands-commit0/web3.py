import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Union
from aiohttp import ClientResponse, ClientSession, ClientTimeout
from eth_typing import URI
import requests
from web3._utils.async_caching import async_lock
from web3._utils.caching import generate_cache_key
from web3.utils.caching import SimpleCache
logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 10
_session_cache = SimpleCache()
_session_cache_lock = threading.Lock()
_async_session_cache = SimpleCache()
_async_session_cache_lock = threading.Lock()
_async_session_pool = ThreadPoolExecutor(max_workers=1)