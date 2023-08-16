import cv2
import face_recognition

# Load images
extracted = cv2.imread('Face_Matcher\\images\\extract.jpg')
taken = cv2.imread('Face_Matcher\\images\\taken.jpg')

# Convert images to grayscale
gray_extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
gray_taken = cv2.cvtColor(taken, cv2.COLOR_BGR2GRAY)

# Save the grayscale images to temporary files
temp_extracted_path = 'temp_extracted.jpg'
temp_taken_path = 'temp_taken.jpg'

cv2.imwrite(temp_extracted_path, gray_extracted)
cv2.imwrite(temp_taken_path, gray_taken)

# Load images into face_recognition
image_extracted = face_recognition.load_image_file(temp_extracted_path)
image_taken = face_recognition.load_image_file(temp_taken_path)

# Find face encodings
face_extracted_encoding = face_recognition.face_encodings(image_extracted)[0]
face_taken_encoding = face_recognition.face_encodings(image_taken)[0]

# Compare face encodings
face_distance = face_recognition.face_distance([face_extracted_encoding], face_taken_encoding)

# Set a threshold for similarity
threshold = 0.6

if face_distance < threshold:
    print("The faces are likely the same person.")
    print(face_distance)
else:
    print("The faces are likely different people.")
    print(face_distance)

# Clean up temporary files
import os
os.remove(temp_extracted_path)
os.remove(temp_taken_path)