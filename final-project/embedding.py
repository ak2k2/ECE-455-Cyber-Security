import numpy as np
import tensorflow as tf
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# If a GPU is available, use it (highly recommended for performance)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define MTCNN module for face detection
mtcnn = MTCNN(image_size=224, margin=0, min_face_size=20, device=device)


# Define Inception Resnet V1 module for face embeddings extraction
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def get_embedding_from_face(image):
    # Detect face and return cropped face
    aligned_face = mtcnn(image)

    # Calculate embedding (unsqueeze to add batch dimension)
    if aligned_face is not None:
        aligned_face = aligned_face.to(device)
        embedding = resnet(aligned_face.unsqueeze(0))

        return embedding.detach().cpu().numpy()


def verify_face(new_embedding, stored_embeddings, threshold=0.8):
    # Compute cosine similarity between the new embedding and each stored embedding
    similarities = cosine_similarity(new_embedding, stored_embeddings)

    # Check if any of the similarities exceed the threshold
    match = any(similarity >= threshold for similarity in similarities[0])

    return match
