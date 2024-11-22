from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, Union, cast
from eth_typing import ChecksumAddress
from eth_utils import is_0x_prefixed, is_hex, is_hex_address
from ens import ENS, AsyncENS
from web3.exceptions import NameNotFound

from web3._utils.validation_utils import is_ens_name
if TYPE_CHECKING:
    from web3 import AsyncWeb3, Web3
    from web3.contract import Contract

class StaticENS:

    def __init__(self, name_addr_pairs: Dict[str, ChecksumAddress]) -> None:
        self.registry = dict(name_addr_pairs)

@contextmanager
def contract_ens_addresses(contract: 'Contract', name_addr_pairs: Dict[str, ChecksumAddress]) -> Iterator[None]:
    """
    Use this context manager to temporarily resolve name/address pairs
    supplied as the argument. For example:

    with contract_ens_addresses(mycontract, [('resolve-as-1s.eth', '0x111...111')]):
        # any contract call or transaction in here would only resolve the above ENS pair
    """
    tmp_ens = contract.w3.ens
    contract.w3.ens = StaticENS(name_addr_pairs)
    try:
        yield
    finally:
        contract.w3.ens = tmp_ens