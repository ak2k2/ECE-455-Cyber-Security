import io
import pickle
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from embedding import get_embedding_from_face, verify_face
from helpers import decrypt_message

# Load the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access and encode the encryption key to bytes
FERNET_KEY = config["encryption"]["key"].encode()


def get_df_from_db(db_path: str):
    conn = sqlite3.connect(db_path)
    USERS = pd.read_sql_query("SELECT * FROM user", conn)
    conn.close()
    return USERS


def get_embedding_by_username(username: str):
    USERS = get_df_from_db("user/users.db")

    user_data = USERS[USERS["username"] == username]
    if user_data.empty:
        print("User not found.")
        return None

    encrypted_embedding = user_data.iloc[0]["embedding"]
    if encrypted_embedding is not None:
        decrypted_embedding = decrypt_message(encrypted_embedding, FERNET_KEY)
        embedding = pickle.loads(decrypted_embedding)
        print(embedding.shape)
        return embedding
    else:
        print("Embedding not found.")
        return None


def get_images_by_username(username: str):
    images = []
    USERS = get_df_from_db("user/users.db")

    print(f"Registered Users: {list(USERS['username'])}")

    user_data = USERS[USERS["username"] == username]
    if user_data.empty:
        print("User not found.")
        return []

    for n in [1, 2, 3]:
        image_column = f"image_{n}"
        if (
            image_column in user_data.columns
            and user_data.iloc[0][image_column] is not None
        ):
            encrypted_image = user_data.iloc[0][image_column]
            decrypted_image = decrypt_message(encrypted_image, FERNET_KEY)

            image_bytes = io.BytesIO(decrypted_image)
            image = Image.open(image_bytes)

            plt.imshow(image)
            plt.axis("off")
            plt.title(f"{username}: Image {n}")
            plt.show()

            images.append(image)
        else:
            print(f"Image {n} not found.")

    return images


def validate_two_factor_login(username: str, test_image):
    embedding = get_embedding_by_username(username)
    if embedding is None:
        print("User not found.")
        return False
    test_embedding = get_embedding_from_face(test_image)
    if test_embedding is None:
        print("No face detected.")
        return False
    if verify_face(test_embedding, embedding):
        print("Login successful.")
        return True
    else:
        print("Login unsuccessful.")
        return False


# # read in misc/aryan.png as bytes
# with open("misc/mallory.png", "rb") as file:
#     attempted_face = file.read()
# attempted_face_bytes = io.BytesIO(attempted_face)
# attempted_face = Image.open(attempted_face_bytes).convert("RGB")

# validate_two_factor_login("armaan", attempted_face)
