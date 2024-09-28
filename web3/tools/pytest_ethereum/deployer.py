from typing import Any, Callable, Dict
from eth_typing import ContractName
from ethpm import Package
from web3.tools.pytest_ethereum.exceptions import DeployerError
from web3.tools.pytest_ethereum.linker import deploy, linker

class Deployer:

    def __init__(self, package: Package) -> None:
        if not isinstance(package, Package):
            raise TypeError(f'Expected a Package object, instead received {type(package)}.')
        self.package = package
        self.strategies: Dict[str, Callable[[Package], Package]] = {}