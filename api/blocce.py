import os
import random
import string
from covert_utils import generate_key, generate_ethereum_address,\
encrypt_message,  decrypt_message, create_transaction, \
w3,sign_transaction, send_transaction

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
        print("Generating addresses for embeding covert infomations ...")
        while len(self.sorted_addresses.keys()) < 256 or len(self.addresses) < 3000:
            self.generate_addresses()
            #print(len(self.sorted_addresses.keys()))
            #print(self.sorted_addresses.keys())
        #print(len(self.addresses))

    def generate_addresses(self):
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
          tx = create_transaction(from_address, to_address)
          tx_signed = sign_transaction(w3, tx, self.addresse_to_pkey[from_address])
          tx_hash = send_transaction(w3, tx_signed)
          tx["hash"] = "0x" + tx_hash.hex()
          transactions.append(tx)
        return transactions 

    def decode_message(self, transactions):
        """Decodes the message from transaction history."""
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
    blocce = Blocce(password)
    blocce.generate_addresses()


    def generate_random_string(length=10):
        # Generate a random string of specified length using letters and digits
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=length))
        return random_string


    # Example usage
    message = "a secret message"
    transactions = blocce.encode_message(message)
    print(transactions)
    print(len(transactions))
    print(blocce.decode_message(transactions))

    #import random
    #import string
    #for i in range(10):
    #    message = generate_random_string(100)
    #    print("Random generated string:", message)
    #    transactions = blocce.encode_message(message)
    #    print(transactions)
    #    print(len(transactions))
    #    print(blocce.decode_message(transactions))

    # Extract column headers from the keys of the first dictionary
    headers = list(transactions[0].keys())
    # Open (or create) the CSV file in write mode
    with open('blocce_transactions.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()     # Write header row using dictionary keys
        writer.writerows(transactions)   # Write all rows in one go



