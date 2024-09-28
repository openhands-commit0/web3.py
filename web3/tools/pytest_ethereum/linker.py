import logging
from typing import Any, Callable, Dict
from eth_typing import ContractName
from eth_utils import to_checksum_address, to_hex
from eth_utils.toolz import assoc_in, curry, pipe
from ethpm import Package
from ethpm.uri import create_latest_block_uri
from web3.tools.pytest_ethereum._utils import create_deployment_data, get_deployment_address, insert_deployment
from web3.tools.pytest_ethereum.exceptions import LinkerError
logger = logging.getLogger('pytest_ethereum.linker')

def deploy(contract_name: str, *args: Any, transaction: Dict[str, Any]=None) -> Callable[..., Package]:
    """
    Return a newly created package and contract address.
    Will deploy the given contract_name, if data exists in package. If
    a deployment is found on the current w3 instance, it will return that deployment
    rather than creating a new instance.
    """
    pass

@curry
def link(contract: ContractName, linked_type: str, package: Package) -> Package:
    """
    Return a new package, created with a new manifest after applying the linked type
    reference to the contract factory.
    """
    pass

@curry
def run_python(callback_fn: Callable[..., None], package: Package) -> Package:
    """
    Return the unmodified package, after performing any user-defined
    callback function on the contracts in the package.
    """
    pass