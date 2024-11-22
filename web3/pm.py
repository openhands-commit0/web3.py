from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Iterable, NamedTuple, Tuple, Type, TypeVar, Union, cast
from eth_typing import URI, Address, ChecksumAddress, ContractName, Manifest
from eth_utils import is_canonical_address, is_checksum_address, to_checksum_address, to_text, to_tuple
from ens import ENS
from ethpm import ASSETS_DIR, Package
from ethpm.exceptions import EthPMException, ManifestValidationError
from ethpm.uri import is_supported_content_addressed_uri, resolve_uri_contents
from ethpm.validation.manifest import validate_manifest_against_schema, validate_raw_manifest_format
from ethpm.validation.package import validate_package_name, validate_package_version
from web3 import Web3
from web3._utils.validation_utils import is_ens_name
from web3.exceptions import InvalidAddress, NameNotFound
from web3.module import Module
T = TypeVar('T')

class ReleaseData(NamedTuple):
    package_name: str
    version: str
    manifest_uri: URI

class ERC1319Registry(ABC):
    """
    The ERC1319Registry class is a base class for all registry implementations
    to inherit from. It defines the methods specified in
    `ERC 1319 <https://github.com/ethereum/EIPs/issues/1319>`__.  All of these
    methods are prefixed with an underscore, since they are not intended to be
    accessed directly, but rather through the methods on ``web3.pm``.
    They are unlikely to change, but must be implemented in a `ERC1319Registry`
    subclass in order to be compatible with the `PM` module. Any custom
    methods (eg. not defined in ERC1319) in a subclass should *not* be
    prefixed with an underscore.

    All of these methods must be implemented in any subclass in order to work
    with `web3.pm.PM`. Any implementation specific logic should be
    handled in a subclass.
    """

    @abstractmethod
    def __init__(self, address: Address, w3: Web3) -> None:
        """
        Initializes the class with the on-chain address of the registry, and a web3
        instance connected to the chain where the registry can be found.

        Must set the following properties...

        * ``self.registry``: A `web3.contract` instance of the target registry.
        * ``self.address``: The address of the target registry.
        * ``self.w3``: The *web3* instance connected to the chain where the
                       registry can be found.
        """
        pass

    @abstractmethod
    def _release(self, package_name: str, version: str, manifest_uri: str) -> bytes:
        """
        Returns the releaseId created by successfully adding a release to the registry.

        * Parameters:
            * ``package_name``: Valid package name according the spec.
            * ``version``: Version identifier string, can conform to
                           any versioning scheme.
            * ``manifest_uri``: URI location of a manifest which details the
                                release contents
        """
        pass

    @abstractmethod
    def _get_package_name(self, package_id: bytes) -> str:
        """
        Returns the package name associated with the given package id, if the
        package id exists on the connected registry.

        * Parameters:
            * ``package_id``: 32 byte package identifier.
        """
        pass

    @abstractmethod
    def _get_all_package_ids(self) -> Iterable[bytes]:
        """
        Returns a tuple containing all of the package ids found on the
        connected registry.
        """
        pass

    @abstractmethod
    def _get_release_id(self, package_name: str, version: str) -> bytes:
        """
        Returns the 32 bytes release id associated with the given
        package name and version, if the release exists on the connected registry.

        * Parameters:
            * ``package_name``: Valid package name according the spec.
            * ``version``: Version identifier string, can conform to
                           any versioning scheme.
        """
        pass

    @abstractmethod
    def _get_all_release_ids(self, package_name: str) -> Iterable[bytes]:
        """
        Returns a tuple containing all of the release ids belonging to the
        given package name, if the package has releases on the connected registry.

        * Parameters:
            * ``package_name``: Valid package name according the spec.
        """
        pass

    @abstractmethod
    def _get_release_data(self, release_id: bytes) -> ReleaseData:
        """
        Returns a tuple containing (package_name, version, manifest_uri) for the
        given release id, if the release exists on the connected registry.

        * Parameters:
            * ``release_id``: 32 byte release identifier.
        """
        pass

    @abstractmethod
    def _generate_release_id(self, package_name: str, version: str) -> bytes:
        """
        Returns the 32 byte release identifier that *would* be associated with the given
        package name and version according to the registry's hashing mechanism.
        The release *does not* have to exist on the connected registry.

        * Parameters:
            * ``package_name``: Valid package name according the spec.
            * ``version``: Version identifier string, can conform to
                           any versioning scheme.
        """
        pass

    @abstractmethod
    def _num_package_ids(self) -> int:
        """
        Returns the number of packages that exist on the connected registry.
        """
        pass

    @abstractmethod
    def _num_release_ids(self, package_name: str) -> int:
        """
        Returns the number of releases found on the connected registry,
        that belong to the given package name.

        * Parameters:
            * ``package_name``: Valid package name according the spec.
        """
        pass

    @classmethod
    @abstractmethod
    def deploy_new_instance(cls: Type[T], w3: Web3) -> T:
        """
        Class method that returns a newly deployed instance of ERC1319Registry.

        * Parameters:
            * ``w3``: Web3 instance on which to deploy the new registry.
        """
        pass
BATCH_SIZE = 100

class SimpleRegistry(ERC1319Registry):
    """
    This class represents an instance of the `Solidity Reference Registry implementation
    <https://github.com/ethpm/solidity-registry>`__.
    """

    def __init__(self, address: ChecksumAddress, w3: Web3) -> None:
        abi = get_simple_registry_manifest()['contractTypes']['PackageRegistry']['abi']
        self.registry = w3.eth.contract(address=address, abi=abi)
        self.address = address
        self.w3 = w3

class PM(Module):
    """
    The PM module will work with any subclass of ``ERC1319Registry``,
    tailored to a particular implementation of
    `ERC1319  <https://github.com/ethereum/EIPs/issues/1319>`__, set as
    its ``registry`` attribute.
    """
    w3: 'Web3'

    def get_package_from_manifest(self, manifest: Manifest) -> Package:
        """
        Returns a `Package <https://github.com/ethpm/py-ethpm/blob/master/ethpm/package.py>`__  # noqa: E501
        instance built with the given manifest.

        * Parameters:
            * ``manifest``: A dict representing a valid manifest
        """
        pass

    def get_package_from_uri(self, manifest_uri: URI) -> Package:
        """
        Returns a `Package <https://github.com/ethpm/py-ethpm/blob/master/ethpm/package.py>`__  # noqa: E501
        instance built with the Manifest stored at the URI.
        If you want to use a specific IPFS backend, set ``ETHPM_IPFS_BACKEND_CLASS``
        to your desired backend. Defaults to Infura IPFS backend.

        * Parameters:
            * ``uri``: Must be a valid content-addressed URI
        """
        pass

    def get_local_package(self, package_name: str, ethpm_dir: Path=None) -> Package:
        """
        Returns a `Package <https://github.com/ethpm/py-ethpm/blob/master/ethpm/package.py>`__  # noqa: E501
        instance built with the Manifest found at the package name in your local ethpm_dir.

        * Parameters:
            * ``package_name``: Must be the name of a package installed locally.
            * ``ethpm_dir``: Path pointing to the target ethpm directory (optional).
        """
        pass

    def set_registry(self, address: Union[Address, ChecksumAddress, ENS]) -> None:
        """
        Sets the current registry used in ``web3.pm`` functions that read/write
        to an on-chain registry. This method accepts checksummed/canonical
        addresses or ENS names. Addresses must point to an on-chain instance
        of an ERC1319 registry implementation.

        To use an ENS domain as the address, make sure a valid ENS instance
        set as ``web3.ens``.

        * Parameters:
            * ``address``: Address of on-chain Registry.
        """
        pass

    def deploy_and_set_registry(self) -> ChecksumAddress:
        """
        Returns the address of a freshly deployed instance of `SimpleRegistry`
        and sets the newly deployed registry as the active registry on
        ``web3.pm.registry``.

        To tie your registry to an ENS name, use web3's ENS module, ie.

        .. code-block:: python

           w3.ens.setup_address(ens_name, w3.pm.registry.address)
        """
        pass

    def release_package(self, package_name: str, version: str, manifest_uri: URI) -> bytes:
        """
        Returns the release id generated by releasing a package on the current registry.
        Requires ``web3.PM`` to have a registry set. Requires
        ``web3.eth.default_account`` to be the registry owner.

        * Parameters:
            * ``package_name``: Must be a valid package name, matching the
                                given manifest.
            * ``version``: Must be a valid package version, matching the given manifest.
            * ``manifest_uri``: Must be a valid content-addressed URI. Currently,
                                only IPFS and Github content-addressed URIs are
                                supported.

        """
        pass

    @to_tuple
    def get_all_package_names(self) -> Iterable[str]:
        """
        Returns a tuple containing all the package names
        available on the current registry.
        """
        pass

    def get_package_count(self) -> int:
        """
        Returns the number of packages available on the current registry.
        """
        pass

    def get_release_count(self, package_name: str) -> int:
        """
        Returns the number of releases of the given package name
        available on the current registry.
        """
        pass

    def get_release_id(self, package_name: str, version: str) -> bytes:
        """
        Returns the 32 byte identifier of a release for the given package
        name and version, if they are available on the current registry.
        """
        pass

    @to_tuple
    def get_all_package_releases(self, package_name: str) -> Iterable[Tuple[str, str]]:
        """
        Returns a tuple of release data (version, manifest_ur) for every release of the
        given package name available on the current registry.
        """
        pass

    def get_release_id_data(self, release_id: bytes) -> ReleaseData:
        """
        Returns ``(package_name, version, manifest_uri)`` associated with the given
        release id, *if* it is available on the current registry.

        * Parameters:
            * ``release_id``: 32 byte release identifier
        """
        pass

    def get_release_data(self, package_name: str, version: str) -> ReleaseData:
        """
        Returns ``(package_name, version, manifest_uri)`` associated with the given
        package name and version, *if* they are published to the currently set registry.

        * Parameters:
            * ``name``: Must be a valid package name.
            * ``version``: Must be a valid package version.
        """
        pass

    def get_package(self, package_name: str, version: str) -> Package:
        """
        Returns a ``Package`` instance, generated by the ``manifest_uri``
        associated with the given package name and version, if they are
        published to the currently set registry.

        * Parameters:
            * ``name``: Must be a valid package name.
            * ``version``: Must be a valid package version.
        """
        pass