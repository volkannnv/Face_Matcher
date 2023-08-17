import cv2
import dlib
import numpy as np

# Load images
image1_path = 'Face_Matcher\\images\\extract.jpg'
image2_path = 'Face_Matcher\\images\\taken.jpg'

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Convert images to RGB format (dlib expects RGB images)
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Initialize face detector and recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('Face_Matcher\\models\\shape_predictor_5_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('Face_Matcher\\models\\dlib_face_recognition_resnet_model_v1.dat')

# Detect faces in images
faces1 = face_detector(image1_rgb)
faces2 = face_detector(image2_rgb)

# Ensure one face is detected in each image
if len(faces1) != 1 or len(faces2) != 1:
    print("Error: Exactly one face should be detected in each image.")
else:
    # Compute face shape and descriptor
    shape1 = shape_predictor(image1_rgb, faces1[0])
    shape2 = shape_predictor(image2_rgb, faces2[0])
    face_descriptor1 = face_recognizer.compute_face_descriptor(image1_rgb, shape1)
    face_descriptor2 = face_recognizer.compute_face_descriptor(image2_rgb, shape2)

    # Compare face descriptors (using Euclidean distance)
    distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))

    # Set a threshold for similarity
    threshold = 0.6

    if distance < threshold:
        print("The faces are likely the same person.")
        print("Distance:", distance)
    else:
        print("The faces are likely different people.")
        print("Distance:", distance)
