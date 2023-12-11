import io
import sqlite3
import urllib.request

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from helpers import decrypt_message

# Downloading the "shape_predictor_68_face_landmarks.dat" model file for dlib


# URL for the pre-trained model
model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# Model file path
model_path = "/mnt/data/shape_predictor_68_face_landmarks.dat.bz2"

# Download the model
urllib.request.urlretrieve(model_url, model_path)

model_path

# Load the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access and encode the encryption key to bytes
FERNET_KEY = config["encryption"]["key"].encode()


def display_user_images(username: str):
    images = []
    # Define the path to the database
    db_path = "user/users.db"

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Load the user table into a pandas DataFrame
    USERS = pd.read_sql_query("SELECT * FROM user", conn)

    # print a list of all the users
    print(f"Registered Users: {list(USERS['username'])}")

    # Close the database connection
    conn.close()

    # Filter the DataFrame for the specific user
    user_data = USERS[USERS["username"] == username]

    if user_data.empty:
        print("User not found.")
        return

    for n in [1, 2, 3]:
        image_column = f"image_{n}"
        if (
            image_column in user_data.columns
            and user_data.iloc[0][image_column] is not None
        ):
            # Decrypt the image data
            encrypted_image = user_data.iloc[0][image_column]
            decrypted_image = decrypt_message(encrypted_image, FERNET_KEY)

            # Convert decrypted data to an image
            image_bytes = io.BytesIO(decrypted_image)
            image = Image.open(image_bytes)

            # # Display the image
            # plt.imshow(image)
            # plt.axis("off")
            # plt.title(f"{username}: Image {n}")
            # plt.show()

            images.append(image)

        else:
            print(f"Image {n} not found.")

    return images


# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)  # Provide the path to the model


def align_and_crop_face(image):
    # Read the image data which is <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x11913B190> type
    # Convert the image to a NumPy array
    image = np.array(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray, 1)

    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)

        # For simplicity, let's use two eye landmarks to align the face
        leftEye = (landmarks.part(36).x, landmarks.part(36).y)
        rightEye = (landmarks.part(45).x, landmarks.part(45).y)

        # Compute the angle between the eye centroids
        dY = rightEye[1] - leftEye[1]
        dX = rightEye[0] - leftEye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Compute the center of the two eyes
        centerX = (leftEye[0] + rightEye[0]) // 2
        centerY = (leftEye[1] + rightEye[1]) // 2

        # Align the image
        M = cv2.getRotationMatrix2D((centerX, centerY), angle, 1)
        aligned = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC
        )

        # Crop the aligned image (you can define the crop size as needed)
        cropped = aligned[face.top() : face.bottom(), face.left() : face.right()]

        # Histogram Equalization for enhancing the image
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(cropped)

        return equalized

    # Return None if no faces are detected
    return None


def get_face_by_username(username: str):
    images = display_user_images(username)

    return images


if __name__ == "__main__":
    images = get_face_by_username("alice")
    for image in images:
        plt.imshow(image)
        plt.axis("off")
        plt.show()
