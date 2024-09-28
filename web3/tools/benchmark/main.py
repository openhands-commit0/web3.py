import argparse
import asyncio
from collections import defaultdict
import logging
import sys
import timeit
from typing import Any, Callable, Dict, Union
from web3 import AsyncHTTPProvider, AsyncWeb3, HTTPProvider, Web3
from web3.middleware import async_buffered_gas_estimate_middleware, async_gas_price_strategy_middleware, buffered_gas_estimate_middleware, gas_price_strategy_middleware
from web3.tools.benchmark.node import GethBenchmarkFixture
from web3.tools.benchmark.reporting import print_entry, print_footer, print_header
from web3.tools.benchmark.utils import wait_for_aiohttp, wait_for_http
from web3.types import Wei
KEYFILE_PW = 'web3py-test'
parser = argparse.ArgumentParser()
parser.add_argument('--num-calls', type=int, default=10, help='The number of RPC calls to make')
if __name__ == '__main__':
    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    main(logger, args.num_calls)