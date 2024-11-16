import cv2
import numpy as np

# Load the target (reference) image of the suspect
target_image_path = "Lab_6/Task3.png"  # Path to the suspect's image
target_img = cv2.imread(target_image_path)
gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect face in the target image
target_faces = face_cascade.detectMultiScale(gray_target, scaleFactor=1.1, minNeighbors=5)
if len(target_faces) == 0:
    print("No face detected in the target image.")
    exit()
else:
    print("No face detected in the target image.")
    x, y, w, h = target_faces[0]
    target_face = gray_target[y:y+h, x:x+w]

# Load the video file to check
video_path = "Lab_6/Task3.mp4"  # Path to the input video
cap = cv2.VideoCapture(video_path)

# Variables to track the best match
best_match_score = float('inf')
best_frame_index = -1
best_face_coords = None
frame_index = 0

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends or no frame is returned

    frame_index += 1
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Compare each detected face with the target face
    for (x, y, w, h) in faces:
        detected_face = gray_frame[y:y+h, x:x+w]

        # Resize face regions to a common size for comparison
        resized_target_face = cv2.resize(target_face, (100, 100))
        resized_detected_face = cv2.resize(detected_face, (100, 100))

        # Compute the similarity score (using Mean Squared Error)
        score = np.sum((resized_target_face - resized_detected_face) ** 2) / (100 * 100)

        # If the score is lower, update the best match
        if score < best_match_score:
            best_match_score = score
            best_frame_index = frame_index
            best_face_coords = (x, y, w, h)
            best_match_frame = frame.copy()  # Store the frame with the best match

# Display the best match
if best_match_frame is not None:
    x, y, w, h = best_face_coords
    cv2.rectangle(best_match_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(best_match_frame, f"Best Match - Frame {best_frame_index}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    print(f"Best match found at frame {best_frame_index} with score {best_match_score}")
    
    # Show the best match frame
    cv2.imshow("Best Match", best_match_frame)
    cv2.waitKey(0)
else:
    print("No matching faces found in the video.")

# Release video resources and close windows
cap.release()
cv2.destroyAllWindows()