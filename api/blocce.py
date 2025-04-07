import os
import random
import string
from covert_utils import generate_key, generate_ethereum_address,\
encrypt_message,  decrypt_message, create_transaction, \
w3,sign_transaction, send_transaction

from prefund_accounts import genesis_accounts

import datetime

class Blocce:
    def __init__(self, password, salt=None):
        self.start_indicator = "λ"   # Start indicator for the encoded message
        self.password = password
        self.key, self.salt = generate_key(password, salt=None)
        #print(self.key, self.salt)
        self.pkeys = []
        self.addresses = []
        self.addresse_to_pkey = {}
        self.sorted_addresses = {}
        self.inverse_sorted_addresses = {}
        self.funding_accounts = genesis_accounts
        print("Generating addresses for embeding covert infomations ...")
        while len(self.sorted_addresses.keys()) < 256 or len(self.addresses) < 3000:
            self.generate_addresses()
            #print(len(self.sorted_addresses.keys()))
            #print(self.sorted_addresses.keys())
        #print(len(self.addresses))
        
    def generate_addresses(self):
        tx_hashes = []
        fund_transactions = []
        for i in range(1000):
            account = generate_ethereum_address()
            address = account['address']
            self.addresses.append(address)
            self.addresse_to_pkey[address] = account['private_key']
            last_byte = bytes.fromhex(address[-2:])
            if last_byte not in self.sorted_addresses:
                self.sorted_addresses[last_byte] = []
            self.sorted_addresses[last_byte].append(address)
            self.inverse_sorted_addresses[address] = last_byte
            # fund generated account with random ammout 
            fund_account = random.choices(self.funding_accounts)[0]
            value = random.uniform(0.5, 1.5)
            tx = create_transaction(fund_account["address"], address, value_in_ether=value)
            # send this funding transaction
            tx_signed = sign_transaction(w3, tx, fund_account["private_key"])
            #tx_hash = send_transaction(w3, tx_signed)
            #tx["hash"] = "0x" + tx_hash.hex()
            #print(tx)
            fund_transactions.append(tx)
            #tx_hashes.append(tx_hash) 
            # wait for transaction receipt 
            #tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            #print(f"Transaction successful! Hash: {tx_hash.hex()}")
            #tx = w3.eth.get_transaction(tx_hash)
            #block = w3.eth.get_block(tx.blockNumber)
            ## Extract the Unix timestamp from the block data
            #timestamp = block.timestamp
            ## Convert the timestamp to a human-readable UTC format
            #readable_time = datetime.datetime.utcfromtimestamp(timestamp)
            #print(f"Transaction Timestamp (UTC): {readable_time}")
        return fund_transactions


    def generate_transactions(self, message):
        """Generates addresses with least significant byte matching the cipher."""
        addresses = []
        for char in cipher:
            bit = bin(ord(char))[-1]  # Get the least significant bit of the character
            while True:
                private_key = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
                public_key = ''.join(random.choices(string.ascii_letters + string.digits, k=64))
                address = ''.join(random.choices(string.ascii_letters + string.digits, k=34))
                if address[-1] == bit:  # Ensure the address matches the bit
                    addresses.append((private_key, public_key, address))
                    break
        return addresses

    def encode_message(self, message):
        """Encodes the message with the secret key."""
        cipher = encrypt_message(self.start_indicator + message, self.key, self.salt)
        transactions = []
        for c in cipher:
            b = c.to_bytes()
            from_address = random.choices(self.sorted_addresses[b])[0]
            to_address = random.choices(self.addresses)[0]
            value = random.uniform(0.001, 0.010)
            tx = create_transaction(from_address, to_address, value_in_ether=value)
            tx_signed = sign_transaction(w3, tx, self.addresse_to_pkey[from_address])
            # send all transactions in order
            #tx_hash = send_transaction(w3, tx_signed)
            #tx["hash"] = "0x" + tx_hash.hex()
            # wait for transaction receipt 
            #tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            #print(f"Transaction successful! Hash: {tx_hash.hex()}")
            #tx = w3.eth.get_transaction(tx_hash)
            #block = w3.eth.get_block(tx.blockNumber)
            ## Extract the Unix timestamp from the block data
            #timestamp = block.timestamp
            ## Convert the timestamp to a human-readable UTC format
            #readable_time = datetime.datetime.utcfromtimestamp(timestamp)
            #print(f"Transaction Timestamp (UTC): {readable_time}")
            transactions.append(tx)
        return transactions 

    def decode_message(self, transactions):
        """Decodes the message from transaction history."""
        # assume all transactions in time reversed order
        cipher_bytes = bytes()
        for t in transactions:
            address = t["from"]
            byte = self.inverse_sorted_addresses[address]
            cipher_bytes+=byte
        recovered_message = decrypt_message(cipher_bytes, self.key)
        if recovered_message[0] == self.start_indicator:
            return recovered_message[1:]
        else:
            print("No start indicator found")
            return None



import csv

if __name__ == "__main__":
    # Example usage:
    password = "common_password"

    def save_transactions_to_csv(transactions, fname):
        # Extract column headers from the keys of the first dictionary
        headers = list(transactions[0].keys())
        # Open (or create) the CSV file in write mode
        with open(fname, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()     # Write header row using dictionary keys
            writer.writerows(transactions)   # Write all rows in one go

    import nltk
    nltk.download('words')
    from nltk.corpus import words
    def generate_random_words(num_words=1):
        word_list = words.words()
        # random.sample selects unique words; adjust if you want duplicates
        return random.sample(word_list, num_words)

    print(generate_random_words())
    blocce = Blocce(password)
    fund_txs = blocce.generate_addresses()
    save_transactions_to_csv(fund_txs, "fund_transactions.csv")

    # Example usage
    covert_txs = []
    for i in range(100):
        message = generate_random_words()[0]
        print("Random generated string:", message)
        transactions = blocce.encode_message(message)
        covert_txs += transactions
        print(len(transactions))
        print(blocce.decode_message(transactions))
    save_transactions_to_csv(covert_txs, "blocce_transactions.csv")


