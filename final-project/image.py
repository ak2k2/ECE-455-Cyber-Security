import io
import sqlite3

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mtcnn import MTCNN
from PIL import Image

from helpers import decrypt_message

# Initialize MTCNN
detector = MTCNN()

# Load the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access and encode the encryption key to bytes
FERNET_KEY = config["encryption"]["key"].encode()


def get_face_from_db(username: str):
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


def align_and_crop_face(image, face_height_multiplier=2.0):
    # Convert the PIL Image to a NumPy array
    image = np.array(image)

    # Detect faces in the image
    faces = detector.detect_faces(image)

    for face in faces:
        # Get facial landmarks
        keypoints = face["keypoints"]
        leftEye = keypoints["left_eye"]
        rightEye = keypoints["right_eye"]

        # Compute the angle between the eye centroids
        dY = rightEye[1] - leftEye[1]
        dX = rightEye[0] - leftEye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Compute the center of the two eyes
        eye_center_x = (leftEye[0] + rightEye[0]) // 2
        eye_center_y = (leftEye[1] + rightEye[1]) // 2

        # Align the image
        M = cv2.getRotationMatrix2D((eye_center_x, eye_center_y), angle, 1)
        aligned = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC
        )

        # Get the bounding box for cropping
        x, y, width, height = face["box"]

        # Find the center of the face
        face_center_x = x + width // 2
        face_center_y = y + height // 2

        # Define the crop size
        crop_size = int(height * face_height_multiplier)

        # Calculate the cropping coordinates (ensuring they are within the image bounds)
        x_start = max(face_center_x - crop_size // 2, 0)
        y_start = max(face_center_y - crop_size // 2, 0)
        x_end = min(face_center_x + crop_size // 2, image.shape[1])
        y_end = min(face_center_y + crop_size // 2, image.shape[0])

        # Crop the aligned image
        cropped = aligned[y_start:y_end, x_start:x_end]
        final_size = 224

        # Determine the larger dimension (height or width) and crop to make it square
        crop_height, crop_width = cropped.shape[:2]
        if crop_height > crop_width:
            start = (crop_height - crop_width) // 2
            square_crop = cropped[start : start + crop_width, :]
        else:
            start = (crop_width - crop_height) // 2
            square_crop = cropped[:, start : start + crop_height]

        # Resize the square crop to the desired final size (224x224)
        resized_crop = cv2.resize(square_crop, (final_size, final_size))

        return resized_crop

    # Return None if no faces are detected
    return None


def get_face_by_username(username: str):
    images = get_face_from_db(username)
    if images is None:
        print("No images found.")
        return
    images = [align_and_crop_face(image) for image in images]
    for image in images:
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    return images
