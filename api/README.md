# setup local testnet  

## 
Makesure docker and kurtosis is installed, then start local testnet by:
'''
kurtosis --enclave local-eth-testnet run github.com/ethpandaops/ethereum-package
'''

Make a virtual env:
'''
python3 -m local_testnet_venv ./local_testnet_venv
source ./local_testnet_venv/bin/activate
pip3 install -r local_testnet_venv.txt 
'''

Enter working directory:
'''
cd local_testnet
'''

Check docker port for geth and modify:
'''
# Replace the URL with your node's address
w3 = Web3(Web3.HTTPProvider('http://localhost:<$RPC_PORT>'))
'''
Then test if network is alive and transaction is possible:
'''
python test_local_testnet_alive.py
python test_local_testnet_transaction.py
'''
shutdown testnet by:
'''
kurtosis clean -a
'''
