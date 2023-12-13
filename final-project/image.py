import cv2
import numpy as np
import torch
from mtcnn import MTCNN
from PIL import Image

from helpers import decrypt_message

# If a GPU is available, use it (highly recommended for performance)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define MTCNN module for face detection
detector = MTCNN(device=device)


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
