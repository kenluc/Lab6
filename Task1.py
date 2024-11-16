import cv2
import numpy as np

# Load the video
video_path = "Lab_6/Task1_4.mp4"
cap = cv2.VideoCapture(video_path)

# Define the background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Define parameters for object detection and tracking
min_contour_area = 1000  # Minimum area to consider as a person (adjust as needed)
centroid_history = {}  # Store previous centroid positions for each person

# Open a video writer to save the output with bounding boxes
output_path = "Lab_6/Output_Task1.mp4"  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame-by-frame
frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to identify moving objects
    fg_mask = bg_subtractor.apply(frame)

    # Remove shadows and noise using thresholding
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours by area, descending
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:2]  # Top 3 largest contours

    detected_centroids = []

    # Loop through the top 3 contours
    for i, contour in enumerate(filtered_contours):
        # Get bounding box and centroid for each detected contour
        x, y, w, h = cv2.boundingRect(contour)
        centroid = (int(x + w / 2), int(y + h / 2))

        # Save current frame centroids
        detected_centroids.append(centroid)

        # Draw bounding box and label as "Person" on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Person ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Track detected centroids over frames to maintain individual tracking
    for i, centroid in enumerate(detected_centroids):
        # If the centroid is new, initialize it
        if i not in centroid_history:
            centroid_history[i] = centroid
        else:
            # Calculate movement if we have a previous centroid position
            previous_centroid = centroid_history[i]
            movement = np.array(centroid) - np.array(previous_centroid)
            print(f"Frame {frame_index}: Person {i+1} moved by {movement}")

            # Update the centroid history for this person
            centroid_history[i] = centroid

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with tracking and bounding boxes (optional)
    cv2.imshow("Person Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Tracking complete. Output video saved.")