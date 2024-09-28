import collections
import math
import operator
from typing import Iterable, Sequence, Tuple
from eth_typing import ChecksumAddress
from eth_utils import to_tuple
from eth_utils.toolz import curry, groupby, sliding_window
from hexbytes import HexBytes
from web3 import Web3
from web3._utils.math import percentile
from web3.exceptions import InsufficientData, Web3ValidationError
from web3.types import BlockNumber, GasPriceStrategy, TxParams, Wei
MinerData = collections.namedtuple('MinerData', ['miner', 'num_blocks', 'min_gas_price', 'low_percentile_gas_price'])
Probability = collections.namedtuple('Probability', ['gas_price', 'prob'])

@to_tuple
def _compute_probabilities(miner_data: Iterable[MinerData], wait_blocks: int, sample_size: int) -> Iterable[Probability]:
    """
    Computes the probabilities that a txn will be accepted at each of the gas
    prices accepted by the miners.
    """
    pass

def _compute_gas_price(probabilities: Sequence[Probability], desired_probability: float) -> Wei:
    """
    Given a sorted range of ``Probability`` named-tuples returns a gas price
    computed based on where the ``desired_probability`` would fall within the
    range.

    :param probabilities: An iterable of `Probability` named-tuples
        sorted in reverse order.
    :param desired_probability: An floating point representation of the desired
        probability. (e.g. ``85% -> 0.85``)
    """
    pass

@curry
def construct_time_based_gas_price_strategy(max_wait_seconds: int, sample_size: int=120, probability: int=98, weighted: bool=False) -> GasPriceStrategy:
    """
    A gas pricing strategy that uses recently mined block data to derive a gas
    price for which a transaction is likely to be mined within X seconds with
    probability P. If the weighted kwarg is True, more recent block
    times will be more heavily weighted.

    :param max_wait_seconds: The desired maximum number of seconds the
        transaction should take to mine.
    :param sample_size: The number of recent blocks to sample
    :param probability: An integer representation of the desired probability
        that the transaction will be mined within ``max_wait_seconds``.  0 means 0%
        and 100 means 100%.
    """
    pass
fast_gas_price_strategy = construct_time_based_gas_price_strategy(max_wait_seconds=60, sample_size=120)
medium_gas_price_strategy = construct_time_based_gas_price_strategy(max_wait_seconds=600, sample_size=120)
slow_gas_price_strategy = construct_time_based_gas_price_strategy(max_wait_seconds=60 * 60, sample_size=120)
glacial_gas_price_strategy = construct_time_based_gas_price_strategy(max_wait_seconds=24 * 60 * 60, sample_size=720)