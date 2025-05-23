from web3 import Web3
from eth_account import Account
import secrets

import datetime

# Generate a new Ethereum wallet (for demonstration purposes)
def generate_new_wallet(priv=None):
    if priv is None:
        priv = secrets.token_hex(32)
    private_key = "0x" + priv
    account = Account.from_key(private_key)
    return {
        'private_key': private_key,
        'address': account.address
    }

# Connect to an Ethereum node
def connect_to_ethereum(provider_url):
    return Web3(Web3.HTTPProvider(provider_url))

# Send an Ethereum transaction
def send_ethereum_transaction(w3, sender_private_key, recipient_address, amount_in_ether, data=b''):
    # Get the sender's address from the private key
    account = Account.from_key(sender_private_key)
    sender_address = account.address

    # Get the current nonce for the sender
    nonce = w3.eth.get_transaction_count(sender_address)

    # Get the current gas price estimates
    gas_price = w3.eth.gas_price

    # Build the transaction
    transaction = {
        'from': sender_address,
        'to': recipient_address,
        'value': w3.to_wei(amount_in_ether, 'ether'),
        'nonce': nonce,
        'gas': 21000,  # Standard gas limit for a simple ETH transfer
        'gasPrice': gas_price,
        'data': data,
        'chainId': w3.eth.chain_id
    }
    
    #return transaction
    # Sign the transaction
    signed_tx = w3.eth.account.sign_transaction(transaction, sender_private_key)
    # Send the transaction
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    # Wait for the transaction to be mined
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Transaction successful! Hash: {tx_hash.hex()}")
    print(tx_receipt)

    tx = w3.eth.get_transaction(tx_hash)
    block = w3.eth.get_block(tx.blockNumber)
    # Extract the Unix timestamp from the block data
    timestamp = block.timestamp
    # Convert the timestamp to a human-readable UTC format
    readable_time = datetime.datetime.utcfromtimestamp(timestamp)
    print(f"Transaction Timestamp (UTC): {readable_time}")

    return {
        'transaction_hash': tx_hash.hex(),
        'status': 'success' if tx_receipt.status == 1 else 'failed'
    }


#from keys import infura_sepolia_api
# Example usage
def testnet_transaction():
    # Replace with your Ethereum node provider URL (Infura, QuickNode, etc.)
    #provider_url = infura_sepolia_api
    # Connect to Ethereum
    #w3 = connect_to_ethereum(provider_url)
    w3 = connect_to_ethereum('http://localhost:51851')
   
    # Generate or use existing wallets
    sender_wallet = generate_new_wallet('bcdf20249abf0ed6d944c0288fad489e33f66b3960d9e6229c1cd214ed3bbe31')
    receiver_wallet = generate_new_wallet('53321db7c1e331d93a11a41d16f004d7ff63972ec8ec7c25db329728ceeb1710')

    # Send transaction
    # Note: You need to fund the sender wallet before sending a transaction
    # Mint eth from faucet is too slow and the amount is too little for many
    # covert communication algorithms, thus we will not actually send the 
    # transaction to testnet
    result = send_ethereum_transaction(
        w3,
        sender_wallet['private_key'],
        receiver_wallet["address"],
        0.1,# Amount in ETH convert to wei
        "0x", # Hex inputs/data
    )
    print(result)
    print(f"Transaction sent: {result['transaction_hash']}")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    testnet_transaction()
