from typing import Any, Dict
from eth_abi import abi
from eth_typing import URI
from web3._utils.request import async_get_json_from_client_response, async_get_response_from_get_request, async_get_response_from_post_request
from web3._utils.type_conversion import to_bytes_if_hex, to_hex_if_bytes
from web3.exceptions import MultipleFailedRequests, Web3ValidationError
from web3.types import TxParams