from web3 import Web3

# Replace the URL with your node's address
w3 = Web3(Web3.HTTPProvider('http://localhost:51851'))

if w3.is_connected():
    print("Connected to the Ethereum node!")
    # For example, fetch the latest block:
    latest_block = w3.eth.get_block('latest')
    print("Latest Block:", latest_block)
else:
    print("Connection failed!")

