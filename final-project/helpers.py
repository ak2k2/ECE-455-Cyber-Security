import yaml
from cryptography.fernet import Fernet


def get_config(path="config.yaml"):
    # Load the YAML file
    with open(path, "r") as file:
        config = yaml.safe_load(file)


def encrypt_message(message, key):
    """
    Encrypts a message using the provided key.
    """
    f = Fernet(key)
    encrypted_message = f.encrypt(message)
    return encrypted_message


def decrypt_message(encrypted_message, key):
    """
    Decrypts an encrypted message using the provided key.
    """
    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message)
    return decrypted_message


def generate_key():
    """
    Generates a Fernet key.
    """
    return Fernet.generate_key()


print(generate_key())
