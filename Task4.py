
import cv2
import numpy as np

# Load the video
video_path = "Lab_6/Task1_4.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Define parameters for counting
entry_line_position = 250  # Y-coordinate of the entry/exit line in the frame
line_thickness = 2
people_entered = 0
people_exited = 0
min_contour_area = 200  # Minimum contour area to filter out noise

# Video writer setup to save output
output_path = "Lab_6\Task4_Output.mp4"  # Specify output file path
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Variables to store each person's position history
object_id = 0
object_tracks = {}

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

    # Draw entry/exit line
    cv2.line(frame, (0, entry_line_position), (frame.shape[1], entry_line_position), (0, 255, 0), line_thickness)

    # Process each contour
    new_tracks = {}
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Get bounding box and centroid
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            # Track movement with unique object ID
            matched = False
            for oid, track in object_tracks.items():
                previous_cx, previous_cy = track["centroid"]
                
                # Match if the centroid is close to the previous frame's centroid
                if abs(cx - previous_cx) < 50 and abs(cy - previous_cy) < 50:
                    matched = True
                    new_tracks[oid] = {"centroid": (cx, cy), "counted": track["counted"]}
                    
                    # Check direction of movement to update entry/exit count
                    if not track["counted"]:
                        if previous_cy < entry_line_position and cy >= entry_line_position:
                            people_entered += 1
                            new_tracks[oid]["counted"] = "entered"
                            print(f"Person entered - Total entered: {people_entered}")
                        elif previous_cy > entry_line_position and cy <= entry_line_position:
                            people_exited += 1
                            new_tracks[oid]["counted"] = "exited"
                            print(f"Person exited - Total exited: {people_exited}")
                    break
            
            # If not matched, create a new track
            if not matched:
                new_tracks[object_id] = {"centroid": (cx, cy), "counted": False}
                object_id += 1

            # Draw bounding box and centroid
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Update object tracks with the new positions
    object_tracks = new_tracks

    # Display counts on frame
    cv2.putText(frame, f"Entered: {people_entered}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Exited: {people_exited}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Write the frame to the output video file
    out.write(frame)

    # Show the frame with annotations
    cv2.imshow("People Counting", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Ensure the video file is saved correctly
cv2.destroyAllWindows()

print(f"Final Count - Entered: {people_entered}, Exited: {people_exited}")