from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from eth_typing.encoding import HexStr
from eth_typing.evm import ChecksumAddress
from hexbytes.main import HexBytes
from web3._utils.compat import Protocol
from web3._utils.decorators import deprecate_method
from web3._utils.miner import make_dag, set_etherbase, set_extra, set_gas_price, start, start_auto_dag, stop, stop_auto_dag
from web3._utils.rpc_abi import RPC
from web3.method import DeprecatedMethod, Method, default_root_munger
from web3.module import Module
from web3.types import EnodeURI, GethWallet, NodeInfo, Peer, TxParams, TxPoolContent, TxPoolInspect, TxPoolStatus

class UnlockAccountWrapper(Protocol):

    def __call__(self, account: ChecksumAddress, passphrase: str, duration: Optional[int]=None) -> bool:
        pass
GETH_PERSONAL_DEPRECATION_MSG = 'Geth now uses `clef` for account and key management. This method will be removed in web3.py `v7`.'

class GethPersonal(Module):
    """
    https://geth.ethereum.org/docs/interacting-with-geth/rpc/ns-personal
    """
    is_async = False
    _ec_recover: Method[Callable[[str, HexStr], ChecksumAddress]] = Method(RPC.personal_ecRecover, mungers=[default_root_munger])
    ec_recover = DeprecatedMethod(_ec_recover, msg=GETH_PERSONAL_DEPRECATION_MSG)
    _import_raw_key: Method[Callable[[str, str], ChecksumAddress]] = Method(RPC.personal_importRawKey, mungers=[default_root_munger])
    import_raw_key = DeprecatedMethod(_import_raw_key, msg=GETH_PERSONAL_DEPRECATION_MSG)
    _list_accounts: Method[Callable[[], List[ChecksumAddress]]] = Method(RPC.personal_listAccounts, is_property=True)
    list_accounts = DeprecatedMethod(_list_accounts, msg=GETH_PERSONAL_DEPRECATION_MSG)
    _list_wallets: Method[Callable[[], List[GethWallet]]] = Method(RPC.personal_listWallets, is_property=True)
    list_wallets = DeprecatedMethod(_list_wallets, msg=GETH_PERSONAL_DEPRECATION_MSG)
    _send_transaction: Method[Callable[[TxParams, str], HexBytes]] = Method(RPC.personal_sendTransaction, mungers=[default_root_munger])
    send_transaction = DeprecatedMethod(_send_transaction, msg=GETH_PERSONAL_DEPRECATION_MSG)
    _sign: Method[Callable[[str, ChecksumAddress, Optional[str]], HexStr]] = Method(RPC.personal_sign, mungers=[default_root_munger])
    sign = DeprecatedMethod(_sign, msg=GETH_PERSONAL_DEPRECATION_MSG)
    _sign_typed_data: Method[Callable[[Dict[str, Any], ChecksumAddress, str], HexStr]] = Method(RPC.personal_signTypedData, mungers=[default_root_munger])
    sign_typed_data = DeprecatedMethod(_sign_typed_data, msg=GETH_PERSONAL_DEPRECATION_MSG)
    _new_account: Method[Callable[[str], ChecksumAddress]] = Method(RPC.personal_newAccount, mungers=[default_root_munger])
    new_account = DeprecatedMethod(_new_account, msg=GETH_PERSONAL_DEPRECATION_MSG)
    _lock_account: Method[Callable[[ChecksumAddress], bool]] = Method(RPC.personal_lockAccount, mungers=[default_root_munger])
    lock_account = DeprecatedMethod(_lock_account, msg=GETH_PERSONAL_DEPRECATION_MSG)
    _unlock_account: Method[UnlockAccountWrapper] = Method(RPC.personal_unlockAccount, mungers=[default_root_munger])
    unlock_account = DeprecatedMethod(_unlock_account, msg=GETH_PERSONAL_DEPRECATION_MSG)

class GethTxPool(Module):
    """
    https://geth.ethereum.org/docs/interacting-with-geth/rpc/ns-txpool
    """
    is_async = False
    content: Method[Callable[[], TxPoolContent]] = Method(RPC.txpool_content, is_property=True)
    inspect: Method[Callable[[], TxPoolInspect]] = Method(RPC.txpool_inspect, is_property=True)
    status: Method[Callable[[], TxPoolStatus]] = Method(RPC.txpool_status, is_property=True)

class ServerConnection(Protocol):

    def __call__(self, host: str='localhost', port: int=8546, cors: str='', apis: str='eth,net,web3') -> bool:
        pass

class GethAdmin(Module):
    """
    https://geth.ethereum.org/docs/interacting-with-geth/rpc/ns-admin
    """
    is_async = False
    add_peer: Method[Callable[[EnodeURI], bool]] = Method(RPC.admin_addPeer, mungers=[default_root_munger])
    datadir: Method[Callable[[], str]] = Method(RPC.admin_datadir, is_property=True)
    node_info: Method[Callable[[], NodeInfo]] = Method(RPC.admin_nodeInfo, is_property=True)
    peers: Method[Callable[[], List[Peer]]] = Method(RPC.admin_peers, is_property=True)
    start_http: Method[ServerConnection] = Method(RPC.admin_startHTTP, mungers=[admin_start_params_munger])
    start_ws: Method[ServerConnection] = Method(RPC.admin_startWS, mungers=[admin_start_params_munger])
    stop_http: Method[Callable[[], bool]] = Method(RPC.admin_stopHTTP, is_property=True)
    stop_ws: Method[Callable[[], bool]] = Method(RPC.admin_stopWS, is_property=True)

class GethMiner(Module):
    """
    https://geth.ethereum.org/docs/interacting-with-geth/rpc/ns-miner
    """
    make_dag = make_dag
    set_extra = set_extra
    set_etherbase = set_etherbase
    set_gas_price = set_gas_price
    start = start
    stop = stop
    start_auto_dag = start_auto_dag
    stop_auto_dag = stop_auto_dag

class Geth(Module):
    personal: GethPersonal
    admin: GethAdmin
    txpool: GethTxPool

class AsyncGethTxPool(Module):
    """
    https://geth.ethereum.org/docs/interacting-with-geth/rpc/ns-txpool
    """
    is_async = True
    _content: Method[Callable[[], Awaitable[TxPoolContent]]] = Method(RPC.txpool_content, is_property=True)
    _inspect: Method[Callable[[], Awaitable[TxPoolInspect]]] = Method(RPC.txpool_inspect, is_property=True)
    _status: Method[Callable[[], Awaitable[TxPoolStatus]]] = Method(RPC.txpool_status, is_property=True)

class AsyncGethAdmin(Module):
    """
    https://geth.ethereum.org/docs/interacting-with-geth/rpc/ns-admin
    """
    is_async = True
    _add_peer: Method[Callable[[EnodeURI], Awaitable[bool]]] = Method(RPC.admin_addPeer, mungers=[default_root_munger])
    _datadir: Method[Callable[[], Awaitable[str]]] = Method(RPC.admin_datadir, is_property=True)
    _node_info: Method[Callable[[], Awaitable[NodeInfo]]] = Method(RPC.admin_nodeInfo, is_property=True)
    _peers: Method[Callable[[], Awaitable[List[Peer]]]] = Method(RPC.admin_peers, is_property=True)
    _start_http: Method[Callable[[str, int, str, str], Awaitable[bool]]] = Method(RPC.admin_startHTTP, mungers=[admin_start_params_munger])
    _stop_http: Method[Callable[[], Awaitable[bool]]] = Method(RPC.admin_stopHTTP, is_property=True)
    _start_ws: Method[Callable[[str, int, str, str], Awaitable[bool]]] = Method(RPC.admin_startWS, mungers=[admin_start_params_munger])
    _stop_ws: Method[Callable[[], Awaitable[bool]]] = Method(RPC.admin_stopWS, is_property=True)

class AsyncGethPersonal(Module):
    """
    https://geth.ethereum.org/docs/interacting-with-geth/rpc/ns-personal
    """
    is_async = True
    _ec_recover: Method[Callable[[str, HexStr], Awaitable[ChecksumAddress]]] = Method(RPC.personal_ecRecover, mungers=[default_root_munger])
    _import_raw_key: Method[Callable[[str, str], Awaitable[ChecksumAddress]]] = Method(RPC.personal_importRawKey, mungers=[default_root_munger])
    _list_accounts: Method[Callable[[], Awaitable[List[ChecksumAddress]]]] = Method(RPC.personal_listAccounts, is_property=True)
    _list_wallets: Method[Callable[[], Awaitable[List[GethWallet]]]] = Method(RPC.personal_listWallets, is_property=True)
    _send_transaction: Method[Callable[[TxParams, str], Awaitable[HexBytes]]] = Method(RPC.personal_sendTransaction, mungers=[default_root_munger])
    _sign: Method[Callable[[str, ChecksumAddress, Optional[str]], Awaitable[HexStr]]] = Method(RPC.personal_sign, mungers=[default_root_munger])
    _sign_typed_data: Method[Callable[[Dict[str, Any], ChecksumAddress, str], Awaitable[HexStr]]] = Method(RPC.personal_signTypedData, mungers=[default_root_munger])
    _new_account: Method[Callable[[str], Awaitable[ChecksumAddress]]] = Method(RPC.personal_newAccount, mungers=[default_root_munger])
    _lock_account: Method[Callable[[ChecksumAddress], Awaitable[bool]]] = Method(RPC.personal_lockAccount, mungers=[default_root_munger])
    _unlock_account: Method[Callable[[ChecksumAddress, str, Optional[int]], Awaitable[bool]]] = Method(RPC.personal_unlockAccount, mungers=[default_root_munger])

class AsyncGeth(Module):
    is_async = True
    personal: AsyncGethPersonal
    admin: AsyncGethAdmin
    txpool: AsyncGethTxPool