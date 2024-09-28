from eth_typing import ChecksumAddress, HexAddress, HexStr
ADDRESS_ZERO = HexAddress(HexStr('0x0000000000000000000000000000000000000000'))
CHECKSUM_ADDRESSS_ZERO = ChecksumAddress(ADDRESS_ZERO)
MAX_INT = HexStr('0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
HASH_ZERO = HexStr('0x0000000000000000000000000000000000000000000000000000000000000000')
WEI_PER_ETHER = 1000000000000000000
DYNAMIC_FEE_TXN_PARAMS = ('maxFeePerGas', 'maxPriorityFeePerGas')