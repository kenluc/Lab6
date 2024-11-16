import cv2
import numpy as np

# Load the video
video_path = "Lab_6/Task1.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Define the Region of Interest (ROI) as a rectangle (x, y, width, height)
roi_x, roi_y, roi_w, roi_h = 300, 100, 250, 300  # Reduced width of the ROI (previously 400)
roi = (roi_x, roi_y, roi_w, roi_h)

# Variables for tracking time spent in the ROI
dwell_time = {}  # Stores the dwell time for each tracked object
object_id = 0  # Unique ID for each detected object

# Process each frame in the video
frame_count = 0  # To track the frame index (used for time tracking)
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

    # Draw the ROI on the frame
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # Process each contour
    for contour in contours:
        if cv2.contourArea(contour) > 1500:  # Minimum contour area to filter out noise (adjust as needed)
            # Get bounding box and centroid
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            # Check if the centroid is inside the ROI
            if roi_x <= cx <= roi_x + roi_w and roi_y <= cy <= roi_y + roi_h:
                # If inside the ROI, track the person
                # Assign a unique ID if this is a new object
                if object_id not in dwell_time:
                    dwell_time[object_id] = {'time_in_roi': 0, 'in_roi': False, 'first_frame': frame_count}
                    object_id += 1

                # Update the dwelling time for the object inside the ROI
                for obj_id, obj_info in dwell_time.items():
                    if obj_info['in_roi']:
                        # If person is already in ROI, accumulate time spent in it
                        obj_info['time_in_roi'] += 1 / cap.get(cv2.CAP_PROP_FPS)  # Time spent in seconds

                    # Mark person as inside the ROI
                    if obj_info['first_frame'] == frame_count:
                        dwell_time[obj_id]['in_roi'] = True

            # Draw bounding box and centroid
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Display the total dwelling time for each person
    for obj_id, obj_info in dwell_time.items():
        if obj_info['in_roi']:
            cv2.putText(frame, f"ID:{obj_id} Time: {obj_info['time_in_roi']:.2f}s", 
                        (10, 30 + obj_id * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame with annotations
    cv2.imshow("Dwell Time Tracking", frame)

    # Exit loop on 'q' key press
    frame_count += 1
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the total dwelling time for each person after processing the video
print("\nTotal Dwell Time for each person in ROI:")
for obj_id, obj_info in dwell_time.items():
    print(f"Person ID: {obj_id}, Total Time Spent in ROI: {obj_info['time_in_roi']:.2f} seconds")