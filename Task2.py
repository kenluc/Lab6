

import cv2
import numpy as np

# Load the video
video_path = "Lab_6/Task1.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Define parameters for counting
min_contour_area = 1500  # Minimum contour area to filter out noise (adjust based on video)

# Variables for tracking time and people count at specific intervals
interval_duration = 8  # Interval duration in seconds
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # FPS of the video
frame_count = 0
interval_people_count = {i * interval_duration: 0 for i in range(4)}  # Initializes for 0, 8, 16, 24 seconds, etc.
previous_bounding_boxes = []  # List to store the bounding boxes that have already been counted
object_tracks = {}  # Dictionary to track the positions of each person (bounding box and centroid)

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to detect motion
    fg_mask = bg_subtractor.apply(frame)

    # Remove noise and shadows
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count people in the current frame
    current_frame_people = 0
    current_bounding_boxes = []  # To store bounding boxes detected in the current frame

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Get bounding box and centroid
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            # Check if this bounding box is already counted by comparing with previous centroids
            is_counted = False
            for track_id, track in object_tracks.items():
                prev_cx, prev_cy = track["centroid"]
                # If the centroid is close enough to a previous tracked centroid, we consider it the same person
                if abs(cx - prev_cx) < 50 and abs(cy - prev_cy) < 50:
                    is_counted = True
                    break

            # If not counted, then count this bounding box
            if not is_counted:
                # Draw bounding box around the detected person
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Mark centroid with a small circle
                current_frame_people += 1  # Count the person
                object_tracks[len(object_tracks)] = {"centroid": (cx, cy)}  # Add this person to the tracked objects

    # Increment frame count
    frame_count += 1

    # Calculate the timestamp of the current frame (in seconds)
    timestamp_seconds = frame_count / frame_rate

    # Determine the current interval (e.g., 0-8, 9-16, etc.)
    current_interval = int(timestamp_seconds // interval_duration) * interval_duration

    # Reset count for the new interval
    if current_interval <= 24:  # Assuming video is 27 seconds long
        interval_people_count[current_interval] += current_frame_people

    # Display the current frame with the people count and annotations
    cv2.putText(frame, f"Time: {timestamp_seconds:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the current frame with bounding boxes and annotations
    cv2.imshow("People Counting", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the cumulative people counts for each interval
for interval in sorted(interval_people_count.keys()):
    print(f"Total people count for interval {interval}-{interval + interval_duration} seconds: {interval_people_count[interval]}")