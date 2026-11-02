import Decoder
import Encoder
from time import time
from os import listdir
from datetime import datetime
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding, utils

hashtype = hashes.SHA256()
svpd = padding.PSS(mgf=padding.MGF1(hashtype), salt_length=padding.PSS.MAX_LENGTH)
edpd = padding.OAEP(mgf=padding.MGF1(algorithm=hashtype), algorithm=hashtype, label=None)

with open("Config.txt", "r") as config_file:
    name = config_file.read().strip()
with open(f"Keys/{name}_pri.pem", "rb") as key_file:
    pri_key = serialization.load_pem_private_key(key_file.read(), password=None)
with open(f"Keys/{name}_pub.pem", "rb") as key_file:
    pub_key = serialization.load_pem_public_key(key_file.read())

def get_identity(key):
    return key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
identity = get_identity(pub_key)

def sign(message):
    return pri_key.sign(message, svpd, hashtype)
def verify(message, signature, key):
    try:
        key.verify(signature, message, svpd, hashtype)
        return True
    except Exception:
        return False
def encrypt(message, key):
    data = bytearray()
    keysize = (key.key_size + 7) // 8 - (2 * hashtype.digest_size) - 2
    for i in range(0, len(message), keysize):
        data.extend(key.encrypt(message[i:i+keysize], edpd))
    return data
def decrypt(message):
    data = bytearray()
    keysize = (pri_key.key_size + 7) // 8
    for i in range(0, len(message), keysize):
        data.extend(pri_key.decrypt(message[i:i+keysize], edpd))
    return data

def process(message):
    message_len = int.from_bytes(message[:8], "big")
    signature_len = len(message) - message_len
    try:
        data = decrypt(message[8:][:message_len])
    except ValueError:
        return
    identity_len = int.from_bytes(data[:8], "big")
    sender_identity_len = int.from_bytes(data[8:16], "big")
    data_message_len = len(data) - identity_len - sender_identity_len
    for i in listdir("Keys/"):
        if i.endswith("_pub.pem"):
            with open(f"Keys/{i}", "rb") as key_file:
                sender_pub_key = serialization.load_pem_public_key(key_file.read())
                if get_identity(sender_pub_key) == data[16 + identity_len:][:sender_identity_len]:
                    break
                del sender_pub_key
    signature = message[8 + message_len:][:signature_len]
    try:
        sender_identity = f"Unknown claiming to be {i[:-8]}"
        if verify(data, signature, sender_pub_key):
            sender_identity = i[:-8]
    except NameError:
        sender_identity = "Unknown"
    return (data[16 + identity_len + sender_identity_len:][:data_message_len], sender_identity)
def prepare(message, recipient):
    with open(f"Keys/{recipient}_pub.pem", "rb") as key_file:
        recipient_pub_key = serialization.load_pem_public_key(key_file.read())
    recipient_identity = get_identity(recipient_pub_key)
    message = len(recipient_identity).to_bytes(8, "big") + len(identity).to_bytes(8, "big") + recipient_identity + identity + datetime.fromtimestamp(time()).strftime("%Y-%m-%d %I:%M:%S %p: ").encode("utf-8") + message
    signature = sign(message)
    message = encrypt(message, recipient_pub_key)
    data = len(message).to_bytes(8, "big") + message + signature
    return data

if __name__ == "__main__":
    from sys import argv
    if argv[1] == "0":
        data = prepare(b"Jesus is God!", "Tony Kerr")
        Encoder.send(data)
    elif argv[1] == "1":
        import threading
        t = threading.Thread(target=Decoder.recv, daemon=True)
        t.start()
        while True:
            try:
                message = Decoder.Messages.pop(0)
                print(process(message))
            except IndexError:
                pass

