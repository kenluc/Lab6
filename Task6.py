import cv2
import numpy as np

# Load the video
video_path = "Lab_6/Task6.mp4"
cap = cv2.VideoCapture(video_path)

# Define output video settings
output_path = "Lab_6\Task6_Output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output file
fps = int(cap.get(cv2.CAP_PROP_FPS))      # FPS from the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Define HSV range for black color detection
lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([180, 255, 50], dtype=np.uint8)

# Initialize count for black cars
black_car_count = 0

# Define minimum contour area and aspect ratio to consider as a car
min_contour_area = 2000
min_aspect_ratio = 1.5
max_aspect_ratio = 4.0

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to isolate moving objects
    fg_mask = bg_subtractor.apply(frame)

    # Remove shadows by thresholding the mask
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over contours to detect potential cars
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Get bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Filter based on aspect ratio to avoid tall or narrow objects
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                
                # Only consider regions in the lower part of the frame (optional)
                if y > frame.shape[0] // 3:
                    car_region = frame[y:y+h, x:x+w]

                    # Convert car region to HSV color space
                    hsv_car_region = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)

                    # Create a mask to detect black color in the car region
                    black_mask = cv2.inRange(hsv_car_region, lower_black, upper_black)

                    # If a significant portion of the detected region is black, count it as a black car
                    black_pixels = cv2.countNonZero(black_mask)
                    total_pixels = w * h

                    # Threshold for determining a black car based on pixel count in the region
                    if black_pixels > 0.5 * total_pixels:  # Adjust this threshold if needed
                        black_car_count += 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Black Car", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Write the frame with detections to the output video
    out.write(frame)

# Release video capture, writer, and close windows
cap.release()
out.release()

# Output the total count of black cars
print("Total count of black cars detected:", black_car_count)