import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

# Load images
extracted = cv2.imread('Face_Matcher\\images\\extract.jpg')
taken = cv2.imread('Face_Matcher\\images\\taken.jpg')

# Convert images to RGB format (facenet-pytorch expects RGB images)
extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
taken_rgb = cv2.cvtColor(taken, cv2.COLOR_BGR2RGB)

# Resize images to a suitable size for the model
image_size = (160, 160)
extracted_resized = cv2.resize(extracted_rgb, image_size)
taken_resized = cv2.resize(taken_rgb, image_size)

# Initialize MTCNN for face detection and alignment
mtcnn = MTCNN(select_largest=False)

# Detect and align faces
face_regions_extracted, _ = mtcnn.detect(extracted_resized)
face_regions_taken, _ = mtcnn.detect(taken_resized)

# Extract face regions and convert to PyTorch tensors
face_extracted_tensors = []
face_taken_tensors = []

for face in face_regions_extracted:
    x, y, width, height = map(int, face)  # Convert to integers
    face_extracted_region = extracted_resized[y:y+height, x:x+width]
    face_extracted_tensor = torch.tensor(face_extracted_region, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    face_extracted_tensors.append(face_extracted_tensor)

for face in face_regions_taken:
    x, y, width, height = map(int, face)  # Convert to integers
    face_taken_region = taken_resized[y:y+height, x:x+width]
    face_taken_tensor = torch.tensor(face_taken_region, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    face_taken_tensors.append(face_taken_tensor)

# Load deep learning-based face recognition model (e.g., InceptionResnetV1)
face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()

# Preprocess face images for the model
faces_extracted_aligned = [face_recognizer(face) for face in face_extracted_tensors]
faces_taken_aligned = [face_recognizer(face) for face in face_taken_tensors]

# Perform face recognition
embeddings_extracted = [face.squeeze() for face in faces_extracted_aligned]
embeddings_taken = [face.squeeze() for face in faces_taken_aligned]

# Calculate distance between embeddings
distances = [torch.norm(embedding_extracted - embedding_taken, dim=0) for embedding_extracted, embedding_taken in zip(embeddings_extracted, embeddings_taken)]

# Set a threshold for similarity
threshold = 0.6

for d in distances:
    if d < threshold:
        print("likely the same person.")
        print(distances)
        break
else:
    print("likely a different person.")
    print(distances)