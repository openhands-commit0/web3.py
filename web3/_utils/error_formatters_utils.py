import warnings
from eth_abi import abi
from eth_utils import to_bytes
from web3.exceptions import ContractCustomError, ContractLogicError, ContractPanicError, OffchainLookup, TransactionIndexingInProgress
from web3.types import RPCResponse
SOLIDITY_ERROR_FUNC_SELECTOR = '0x08c379a0'
OFFCHAIN_LOOKUP_FUNC_SELECTOR = '0x556f1830'
OFFCHAIN_LOOKUP_FIELDS = {'sender': 'address', 'urls': 'string[]', 'callData': 'bytes', 'callbackFunction': 'bytes4', 'extraData': 'bytes'}
PANIC_ERROR_FUNC_SELECTOR = '0x4e487b71'
PANIC_ERROR_CODES = {'00': 'Panic error 0x00: Generic compiler inserted panics.', '01': 'Panic error 0x01: Assert evaluates to false.', '11': 'Panic error 0x11: Arithmetic operation results in underflow or overflow.', '12': 'Panic error 0x12: Division by zero.', '21': 'Panic error 0x21: Cannot convert value into an enum type.', '22': 'Panic error 0x12: Storage byte array is incorrectly encoded.', '31': "Panic error 0x31: Call to 'pop()' on an empty array.", '32': 'Panic error 0x32: Array index is out of bounds.', '41': 'Panic error 0x41: Allocation of too much memory or array too large.', '51': 'Panic error 0x51: Call to a zero-initialized variable of internal function type.'}
MISSING_DATA = 'no data'

def _parse_error_with_reverted_prefix(data: str) -> str:
    """
    Parse errors from the data string which begin with the "Reverted" prefix.
    "Reverted", function selector and offset are always the same for revert errors
    """
    pass

def _raise_contract_error(response_error_data: str) -> None:
    """
    Decode response error from data string and raise appropriate exception.

        "Reverted " (prefix may be present in `data`)
        Function selector for Error(string): 08c379a (4 bytes)
        Data offset: 32 (32 bytes)
        String length (32 bytes)
        Reason string (padded, use string length from above to get meaningful part)
    """
    pass

def raise_contract_logic_error_on_revert(response: RPCResponse) -> RPCResponse:
    """
    Revert responses contain an error with the following optional attributes:
        `code` - in this context, used for an unknown edge case when code = '3'
        `message` - error message is passed to the raised exception
        `data` - response error details (str, dict, None)

    See also https://solidity.readthedocs.io/en/v0.6.3/control-structures.html#revert
    """
    pass

def raise_transaction_indexing_error_if_indexing(response: RPCResponse) -> RPCResponse:
    """
    Raise an error if ``eth_getTransactionReceipt`` returns an error indicating that
    transactions are still being indexed.
    """
    pass