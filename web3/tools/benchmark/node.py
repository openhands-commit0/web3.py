import os
import socket
from subprocess import PIPE, Popen, check_output
from tempfile import TemporaryDirectory
from typing import Any, Generator, Sequence
import zipfile
from geth.install import get_executable_path, install_geth
from web3.tools.benchmark.utils import kill_proc_gracefully
GETH_FIXTURE_ZIP = 'geth-1.13.9-fixture.zip'
COINBASE = '0xdc544d1aa88ff8bbd2f2aec754b1f1e99e1812fd'

class GethBenchmarkFixture:

    def __init__(self) -> None:
        self.rpc_port = self._rpc_port()
        self.endpoint_uri = self._endpoint_uri()
        self.geth_binary = self._geth_binary()