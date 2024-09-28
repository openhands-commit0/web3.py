"""
Arguments for the script are:
    -v or --version         Solidity version to be used to compile the contracts. If
                            blank, the script uses the latest hard-coded version
                            specified within the script.

    -f or --filename        If left blank, all .sol files will be compiled and the
                            respective contract data will be generated. Pass in a
                            specific ``.sol`` filename here to compile just one file.


To run the script, you will need the ``py-solc-x`` library for compiling the files
as well as ``black`` for linting. You can install those independently or install the
full ``[dev]`` package extra as shown below.

.. code:: sh

    $ pip install "web3[dev]"

The following example compiles all the contracts and generates their respective
contract data that is used across our test files for the test suites. This data gets
generated within the ``contract_data`` subdirectory within the ``contract_sources``
folder.

.. code-block:: bash

    $ cd ../web3.py/web3/_utils/contract_sources
    $ python compile_contracts.py -v 0.8.17
    Compiling OffchainLookup
    ...
    ...
    reformatted ...

To compile and generate contract data for only one ``.sol`` file, specify using the
filename with the ``-f`` (or ``--filename``) argument flag.

.. code-block:: bash

    $ cd ../web3.py/web3/_utils/contract_sources
    $ python compile_contracts.py -v 0.8.17 -f OffchainLookup.sol
    Compiling OffchainLookup.sol
    reformatted ...
"""
import argparse
import os
import re
from typing import Any, Dict, List
import solcx
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-v', '--version', help='Solidity version for compiling contracts.')
arg_parser.add_argument('-f', '--filename', help='(optional) The filename if only one file is to be compiled - otherwise all .sol files will be compiled at once.')
user_args = arg_parser.parse_args()
LATEST_AVAILABLE_SOLIDITY_VERSION = sorted(solcx.get_compilable_solc_versions())[-1]
user_sol_version = user_args.version
solidity_version = user_sol_version if user_sol_version else LATEST_AVAILABLE_SOLIDITY_VERSION
solcx.install_solc(solidity_version)
solcx.set_solc_version(solidity_version)
all_dot_sol_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.sol')]
user_filename = user_args.filename
files_to_compile = [user_filename] if user_filename else all_dot_sol_files
contracts_in_file = {}
compile_files(files_to_compile)
os.system(f'black {os.getcwd()}')