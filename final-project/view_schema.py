import io
import pickle
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

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


def plot_confusion_matrix():
    USERS = get_df_from_db("user/users.db")

    # dispaly USERS.head() as a pretty table

    # Assume 'all_embeddings' is a list of tuples (username, embedding)
    all_embeddings = [
        (username, get_embedding_by_username(username))
        for username in USERS["username"]
    ]

    # Extract the embeddings and convert them to a 2D NumPy array
    embeddings = [emb[1] for emb in all_embeddings]
    embedding_array = np.vstack(embeddings)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(embedding_array)

    # Apply Multidimensional Scaling (MDS)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_result = mds.fit_transform(
        1 - similarity_matrix
    )  # Convert similarity to dissimilarity

    # Plot the results
    plt.figure(figsize=(10, 8))
    for i, (username, _) in enumerate(all_embeddings):
        plt.scatter(mds_result[i, 0], mds_result[i, 1])
        plt.text(mds_result[i, 0], mds_result[i, 1], username)

    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.title("User Look-Alike Visualization")
    plt.show()


plot_confusion_matrix()
