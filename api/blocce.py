
import os
import random
import string
from covert_utils import generate_key, generate_ethereum_address,\
encrypt_message,  decrypt_message, create_transaction

class Blocce:
    def __init__(self, password, salt=None):
        self.start_indicator = "λ"   # Start indicator for the encoded message
        self.password = password
        self.key, self.salt = generate_key(password, salt=None)
        print(self.key, self.salt)
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
          transactions.append(create_transaction(from_address, to_address))
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




# Example usage:
password = "common_password"
blocce = Blocce(password)
blocce.generate_addresses()

message = "a secret message"
transactions = blocce.encode_message(message)
print(len(transactions))
print(blocce.decode_message(transactions))



message = "a secret message"
transactions = blocce.encode_message(message)
print(len(transactions))
print(blocce.decode_message(transactions))







