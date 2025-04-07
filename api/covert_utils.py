# Install required libraries
# pip install web3 cryptography

import secrets
import binascii
import hashlib
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

import web3
from web3 import Web3
from eth_account import Account
# Connect to an Ethereum node (using Infura as an example)
# Replace with your own Infura project ID or other Ethereum node
#from keys import infura_sepolia_api
#w3 = Web3(Web3.HTTPProvider(infura_sepolia_api))

# Create a local Web3 instance for demonstration
w3 = Web3()
print(f"Using Web3.py version: {web3.__version__}")
print(f"Connected to Ethereum network: {w3.is_connected()}")

def generate_ethereum_address():
    """Generate a random Ethereum address with private key."""
    # For demonstration, we'll create a random address
    #private_key = secrets.token_bytes(32)
    #private_key_hex = binascii.hexlify(private_key).decode('ascii')
    #address = '0x' + secrets.token_hex(20)
    priv = secrets.token_hex(32)
    private_key = "0x" + priv
    account = Account.from_key(private_key)
    return {
        "address": account.address,
        "private_key": private_key
    }


def generate_key(password, salt=None):
    """Generate a cryptographic key from a password."""
    if salt is None:
        salt = os.urandom(16)
    # Use PBKDF2 to derive a key from the password
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, 32)
    return key, salt


def encrypt_message(message, key, salt=None):
    """Encrypt a raw message using AES-256 into cipher"""
    # Generate an initialization vector
    if salt==None:
        iv = os.urandom(16)
    else:
        iv = salt
    # Pad the message
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(message.encode()) + padder.finalize()
    # Create an encryptor
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    # Encrypt the message
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    # Return the IV and ciphertext
    return iv + ciphertext


def decrypt_message(encrypted_data, key):
    """Decrypt cipher into raw message using AES-256"""
    # Extract the IV
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    # Create a decryptor
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    # Decrypt the message
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    # Unpad the message
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    return data.decode()


def encode_message(message, method="base64"):
    """Encode a byte string message using different base methods."""
    if isinstance(message, str):
        message = message.encode()
    if method == "base64":
        return base64.b64encode(message)
    elif method == "hex":
        return binascii.hexlify(message)
    else:
        return message


def decode_message(encoded_message, method="base64"):
    """Decode a message using different methods."""
    if method == "base64":
        return base64.b64decode(encoded_message)
    elif method == "hex":
        return binascii.unhexlify(encoded_message)
    else:
        return encoded_message


def embed_message_in_data(message, obfuscate=True, method="hex", add_padding=True):
    """
    Embed a message in transaction data field.
    """
    # Encode the message
    encoded_message = encode_message(message, method)
    if add_padding:
        # Add random padding before and after the message
        prefix_length = secrets.randbelow(32) + 8  # 8-40 bytes of prefix
        suffix_length = secrets.randbelow(32) + 8  # 8-40 bytes of suffix
        prefix = secrets.token_bytes(prefix_length)
        suffix = secrets.token_bytes(suffix_length)
        # Add markers to identify the message
        start_marker = b"STARTM"
        end_marker = b"ENDM"
        data = prefix + start_marker + encoded_message + end_marker + suffix
    else:
        data = encoded_message
    return data


def extract_message_from_data(data, method="hex"):
    """
    Extract a message from transaction data field.
    """
    try:
        # Find the start and end markers
        start_marker = b"STARTM"
        end_marker = b"ENDM"
        start_pos = data.find(start_marker) + len(start_marker)
        end_pos = data.find(end_marker)
        if start_pos >= len(start_marker) and end_pos > start_pos:
            encoded_message = data[start_pos:end_pos]
            return decode_message(encoded_message, method).decode()
        else:
            # Try to decode the entire data
            return decode_message(data, method).decode()
    except Exception as e:
        print(f"Error extracting message: {e}")
        return None


def create_transaction(from_address, to_address, 
                       value_in_ether=0, gas_price_gwei=50,
                       encrypt=False, encryption_key=None,
                       encoding_method="hex", add_padding=True,
                       gas_limit=None, nonce=None):
    """
    Create a transaction with covert data embedded in the data field.
    """
    # Convert Ether to Wei (1 Ether = 10^18 Wei)
    # Convert Gwei to Wei for gas price (1 Gwei = 10^9 Wei)
    value_in_wei = w3.to_wei(value_in_ether, 'ether')
    #gas_price_wei = w3.eth.gas_price
    gas_price_wei = w3.to_wei(gas_price_gwei, 'gwei') 
    if nonce is None:
        nonce = 0  # In real usage: w3.eth.get_transaction_count(from_address)
    # Determine appropriate gas limit if not specified
    if gas_limit is None:
        # Basic transfer: 21000, with data: more based on data size
        gas_limit = 21000 
    # Create the transaction dictionary
    transaction = {
        'from': from_address,
        'to': to_address,
        'value': value_in_wei,
        'gas': gas_limit,
        'gasPrice': gas_price_wei,
        'nonce': nonce,
        'data': "0x" 
    }
    return transaction




def create_covert_transaction(from_address, to_address, secret_message,
                              value_in_ether=0, gas_price_gwei=50,
                              encrypt=False, encryption_key=None,
                              encoding_method="hex", add_padding=True,
                              gas_limit=None, nonce=None):
    """
    Create a transaction with covert data embedded in the data field.
    """
    # Convert Ether to Wei (1 Ether = 10^18 Wei)
    value_in_wei = w3.to_wei(value_in_ether, 'ether')
    # Convert Gwei to Wei for gas price (1 Gwei = 10^9 Wei)
    gas_price_wei = w3.to_wei(gas_price_gwei, 'gwei')
    # In a real implementation, you would get the nonce from the network
    if nonce is None:
        nonce = 0  # In real usage: w3.eth.get_transaction_count(from_address)
    # Process the message
    if encrypt and encryption_key:
        # Encrypt the message
        prepared_message = encrypt_message(secret_message, encryption_key)
    else:
        prepared_message = secret_message.encode() if isinstance(secret_message, str) else secret_message
    # Embed the message in the data field
    data = embed_message_in_data(prepared_message,
                                obfuscate=True,
                                method=encoding_method,
                                add_padding=add_padding)

    # Determine appropriate gas limit if not specified
    if gas_limit is None:
        # Basic transfer: 21000, with data: more based on data size
        gas_limit = 21000 + (len(data) * 68 if data else 0)

    # Create the transaction dictionary
    transaction = {
        'from': from_address,
        'to': to_address,
        'value': value_in_wei,
        'gas': gas_limit,
        'gasPrice': gas_price_wei,
        'nonce': nonce,
        'data': data.hex() if isinstance(data, bytes) else data
    }

    return transaction


def sign_transaction(w3, transaction, private_key):
    """
    Sign an Ethereum transaction.
    """
    signed_tx = w3.eth.account.sign_transaction(transaction, private_key)
    return signed_tx


def send_transaction(w3, signed_transaction):
    """
    Send a signed transaction to the Ethereum network.
    """
    tx_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
    return tx_hash


def split_message_for_multiple_transactions(message, max_length=100):
    """
    Split a message into multiple pieces for transmission in multiple transactions.
    limit each fragment piece to max_length bytes 
    """
    if isinstance(message, str):
        message = message.encode()
    # Calculate number of pieces
    num_pieces = (len(message) + max_length - 1) // max_length
    # Split the message
    pieces = []
    for i in range(num_pieces):
        start = i * max_length
        end = min((i + 1) * max_length, len(message))
        piece = message[start:end]
        # Add metadata (piece number and total pieces)
        metadata = f"PIECE:{i+1}/{num_pieces}:".encode()
        pieces.append(metadata + piece)
    return pieces


def extract_and_combine_split_messages(transaction_data_list, extract_message_from_data):
    """
    Extract and combine raw ciphers that were split across multiple transactions.
    """
    # Extract messages
    messages = []
    for data in transaction_data_list:
        try:
            message = extract_message_from_data(data)
            if message:
                messages.append(message)
        except:
            continue
    # Parse metadata and combine
    pieces = {}
    total_pieces = 0
    for message in messages:
        if message.startswith("PIECE:"):
            # Extract metadata
            metadata_end = message.find(":", 6)
            if metadata_end > 0:
                piece_info = message[6:metadata_end]
                if "/" in piece_info:
                    piece_num, total = map(int, piece_info.split("/"))
                    total_pieces = total
                    content = message[metadata_end+1:]
                    pieces[piece_num] = content
    # Combine the pieces
    if total_pieces > 0:
        combined = ""
        for i in range(1, total_pieces + 1):
            if i in pieces:
                combined += pieces[i]
        return combined
    return None




