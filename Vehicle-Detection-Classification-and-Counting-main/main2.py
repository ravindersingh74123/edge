# import cv2
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # Initialize DeepSORT
# tracker = DeepSort(max_age=30, n_init=5, max_iou_distance=0.7)

# # Initialize video capture
# cap = cv2.VideoCapture("./Videos/video1.mp4")

# # Video frame properties
# FRAME_WIDTH = 900
# FRAME_HEIGHT = 500

# # Line positions for counting
# line_up = 300
# line_down = 200
# up_limit = 150
# down_limit = 400

# # Counters
# cnt_up = 0
# cnt_down = 0

# # Font for text
# font = cv2.FONT_HERSHEY_SIMPLEX

# # Background subtraction for masking
# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500, varThreshold=50)

# # Optical flow parameters
# prev_gray = None

# # Set to store the IDs of vehicles that have crossed the lines
# crossed_up = set()
# crossed_down = set()

# # Vehicle detection and tracking
# def process_frame(frame):
#     global prev_gray, cnt_up, cnt_down, crossed_up, crossed_down

#     # Resize frame for faster processing
#     frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     if prev_gray is None:
#         prev_gray = gray
#         return frame

#     # Calculate Farneback optical flow
#     flow = cv2.calcOpticalFlowFarneback(
#         prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
#     )
#     prev_gray = gray

#     # Compute magnitude and angle of flow vectors
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     mask = cv2.threshold(mag, 2, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

#     # Morphological operations to clean up motion regions
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Find contours in the mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     detections = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500:  # Filter small blobs
#             x, y, w, h = cv2.boundingRect(cnt)
#             detections.append([[x, y, x + w, y + h], 1.0])  # Format for DeepSORT

#     # Update tracker
#     tracks = tracker.update_tracks(detections, frame=frame)

#     for track in tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue

#         # Get bounding box and track ID
#         bbox = track.to_tlbr()  # top-left and bottom-right coordinates
#         track_id = track.track_id

#         # Calculate object position
#         x1, y1, x2, y2 = [int(i) for i in bbox]
#         cx = (x1 + x2) // 2
#         cy = (y1 + y2) // 2

#         # Draw bounding box and track ID
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
#                     font, 0.5, (255, 255, 255), 1)

#         # Line crossing logic (only count once)
#         if line_down < cy < line_up:
#             if track_id not in crossed_up and y2 > line_up:  # Object moving up
#                 cnt_up += 1
#                 crossed_up.add(track_id)  # Mark this vehicle as counted
#             elif track_id not in crossed_down and y1 < line_down:  # Object moving down
#                 cnt_down += 1
#                 crossed_down.add(track_id)  # Mark this vehicle as counted

#     # Draw lines and display counters
#     cv2.line(frame, (0, line_up), (FRAME_WIDTH, line_up), (255, 0, 0), 2)
#     cv2.line(frame, (0, line_down), (FRAME_WIDTH, line_down), (0, 0, 255), 2)
#     cv2.putText(frame, f"Up: {cnt_up}", (10, 40), font, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"Down: {cnt_down}", (10, 80), font, 1, (0, 255, 0), 2)

#     return frame


# # Process video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process and display frame
#     processed_frame = process_frame(frame)
#     cv2.imshow("Vehicle Tracking with Farneback & DeepSORT", processed_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()












# import cv2
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # Initialize DeepSORT tracker
# tracker = DeepSort(max_age=30, n_init=5, max_iou_distance=0.7)

# # Initialize video capture
# cap = cv2.VideoCapture("./Videos/video1.mp4")

# # Video frame properties
# FRAME_WIDTH = 900
# FRAME_HEIGHT = 500

# # Line positions for counting vehicles
# line_up = 300
# line_down = 200
# up_limit = 150
# down_limit = 400

# # Counters for vehicles crossing lines
# cnt_up = 0
# cnt_down = 0

# # Font for text on frames
# font = cv2.FONT_HERSHEY_SIMPLEX

# # Background subtraction for masking
# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500, varThreshold=50)

# # To store already counted vehicle IDs for up and down crossings
# crossed_up = set()
# crossed_down = set()

# # Vehicle detection and tracking logic
# def process_frame(frame):
#     global cnt_up, cnt_down, crossed_up, crossed_down

#     # Resize frame for faster processing
#     frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

#     # Convert frame to grayscale for optical flow calculation
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Calculate optical flow (if first frame, skip optical flow calculation)
#     if 'prev_gray' not in globals():
#         global prev_gray
#         prev_gray = gray
#         return frame

#     # Calculate Farneback optical flow to track movement
#     flow = cv2.calcOpticalFlowFarneback(
#         prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
#     )
#     prev_gray = gray

#     # Get magnitude and angle of the flow vectors to track movement
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     mask = cv2.threshold(mag, 2, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

#     # Use morphological operations to clean up the detected movement regions
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Find contours from the motion mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     detections = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500:  # Only consider large enough moving objects
#             x, y, w, h = cv2.boundingRect(cnt)
#             detections.append([[x, y, x + w, y + h], 1.0])  # Prepare for DeepSORT

#     # Update DeepSORT tracker with the current frame and detections
#     tracks = tracker.update_tracks(detections, frame=frame)

#     # Process each track
#     for track in tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue

#         bbox = track.to_tlbr()  # Get the bounding box coordinates (top-left, bottom-right)
#         track_id = track.track_id

#         x1, y1, x2, y2 = [int(i) for i in bbox]
#         cx = (x1 + x2) // 2  # Calculate center X of the bounding box
#         cy = (y1 + y2) // 2  # Calculate center Y of the bounding box

#         # Draw bounding box and track ID on the frame
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), font, 0.5, (255, 255, 255), 1)

#         # Line crossing logic: count once for up or down crossing
#         if line_down < cy < line_up:  # Check if object is between the lines
#             if track_id not in crossed_up and cy > line_up:  # Vehicle moving up
#                 cnt_up += 1
#                 crossed_up.add(track_id)  # Mark vehicle as counted for up direction
#             elif track_id not in crossed_down and cy < line_down:  # Vehicle moving down
#                 cnt_down += 1
#                 crossed_down.add(track_id)  # Mark vehicle as counted for down direction

#     # Draw lines for vehicle counting
#     cv2.line(frame, (0, line_up), (FRAME_WIDTH, line_up), (255, 0, 0), 2)  # Blue line for up
#     cv2.line(frame, (0, line_down), (FRAME_WIDTH, line_down), (0, 0, 255), 2)  # Red line for down

#     # Display up/down counts on the frame
#     cv2.putText(frame, f"Up: {cnt_up}", (10, 40), font, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"Down: {cnt_down}", (10, 80), font, 1, (0, 0, 255), 2)

#     return frame


# # Process video frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process and display each frame
#     processed_frame = process_frame(frame)
#     cv2.imshow("Vehicle Tracking with Farneback & DeepSORT", processed_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()













# import cv2
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # Initialize DeepSORT tracker
# tracker = DeepSort(max_age=30, n_init=5, max_iou_distance=0.7)

# # Initialize video capture
# cap = cv2.VideoCapture("./Videos/video1.mp4")

# # Video frame properties
# FRAME_WIDTH = 900
# FRAME_HEIGHT = 500

# # Line positions for counting vehicles
# line_up = 300  # Blue line (Up)
# line_down = 200  # Red line (Down)

# # Counters for vehicles crossing lines
# cnt_up = 0
# cnt_down = 0

# # Font for text on frames
# font = cv2.FONT_HERSHEY_SIMPLEX

# # To store already counted vehicle IDs for up and down crossings
# crossed_up = set()
# crossed_down = set()

# # Vehicle detection and tracking logic
# def process_frame(frame):
#     global cnt_up, cnt_down, crossed_up, crossed_down

#     # Resize frame for faster processing
#     frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

#     # Convert frame to grayscale for optical flow calculation
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Calculate optical flow (if first frame, skip optical flow calculation)
#     if 'prev_gray' not in globals():
#         global prev_gray
#         prev_gray = gray
#         return frame

#     # Calculate Farneback optical flow to track movement
#     flow = cv2.calcOpticalFlowFarneback(
#         prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
#     )
#     prev_gray = gray

#     # Get magnitude and angle of the flow vectors to track movement
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     mask = cv2.threshold(mag, 2, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

#     # Use morphological operations to clean up the detected movement regions
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # Find contours from the motion mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     detections = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500:  # Only consider large enough moving objects
#             x, y, w, h = cv2.boundingRect(cnt)
#             detections.append([[x, y, x + w, y + h], 1.0])  # Prepare for DeepSORT

#     # Update DeepSORT tracker with the current frame and detections
#     tracks = tracker.update_tracks(detections, frame=frame)

#     # Process each track
#     for track in tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue

#         bbox = track.to_tlbr()  # Get the bounding box coordinates (top-left, bottom-right)
#         track_id = track.track_id

#         x1, y1, x2, y2 = [int(i) for i in bbox]
#         cx = (x1 + x2) // 2  # Calculate center X of the bounding box
#         cy = (y1 + y2) // 2  # Calculate center Y of the bounding box

#         # Draw bounding box and track ID on the frame
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), font, 0.5, (255, 255, 255), 1)

#         # Check if the vehicle crosses the red or blue line first
#         if track_id not in crossed_up and track_id not in crossed_down:
#             if cy < line_down and cy > line_up:  # Vehicle is between the two lines
#                 if cy < line_down:  # Crosses the red line first (downward movement)
#                     cnt_down += 1
#                     crossed_down.add(track_id)  # Mark the vehicle as counted for down direction
#                 elif cy > line_up:  # Crosses the blue line first (upward movement)
#                     cnt_up += 1
#                     crossed_up.add(track_id)  # Mark the vehicle as counted for up direction

#     # Draw lines for vehicle counting
#     cv2.line(frame, (0, line_up), (FRAME_WIDTH, line_up), (255, 0, 0), 2)  # Blue line for up
#     cv2.line(frame, (0, line_down), (FRAME_WIDTH, line_down), (0, 0, 255), 2)  # Red line for down

#     # Display up/down counts on the frame
#     cv2.putText(frame, f"Up: {cnt_up}", (10, 40), font, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"Down: {cnt_down}", (10, 80), font, 1, (0, 0, 255), 2)

#     return frame


# # Process video frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process and display each frame
#     processed_frame = process_frame(frame)
#     cv2.imshow("Vehicle Tracking with Farneback & DeepSORT", processed_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# from ultralytics import YOLO

# # Load YOLOv8 model pre-trained on COCO dataset
# model = YOLO("yolov8n.pt")  # You can choose other versions like yolov8s.pt for better accuracy

# # Initialize video capture
# cap = cv2.VideoCapture("./Videos/video1.mp4")

# # Video frame properties
# FRAME_WIDTH = 900
# FRAME_HEIGHT = 500

# # Line positions for counting
# line_up = 300
# line_down = 200

# # Counters
# cnt_up = 0
# cnt_down = 0

# # Font for text
# font = cv2.FONT_HERSHEY_SIMPLEX

# # Set to store the IDs of vehicles that have crossed the lines
# crossed_up = set()
# crossed_down = set()

# def process_frame(frame):
#     global cnt_up, cnt_down, crossed_up, crossed_down

#     # Resize frame for faster processing
#     frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

#     # Perform YOLOv8 object detection
#     results = model(frame, conf=0.4)  # Confidence threshold 0.4
#     detections = results[0].boxes.data.cpu().numpy()  # Extract detections

#     for det in detections:
#         x1, y1, x2, y2, conf, cls = det  # Bounding box and class info
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

#         # Class IDs for vehicles in COCO (e.g., car, truck, bus, etc.)
#         if int(cls) in [2, 3, 5, 7]:  # 2: car, 3: motorcycle, 5: bus, 7: truck
#             # Calculate the center of the bounding box
#             cx = (x1 + x2) // 2
#             cy = (y1 + y2) // 2

#             # Draw bounding box and class
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"ID:{int(cls)}", (x1, y1 - 10), font, 0.5, (255, 255, 255), 1)

#             # Line crossing logic
#             if line_down < cy < line_up:  # Check if the object is between the lines
#                 if cy > line_up and cx not in crossed_up:  # Vehicle crosses the red line upwards
#                     cnt_up += 1
#                     crossed_up.add(cx)
#                 elif cy < line_down and cx not in crossed_down:  # Vehicle crosses the blue line downwards
#                     cnt_down += 1
#                     crossed_down.add(cx)

#     # Draw lines and display counters
#     cv2.line(frame, (0, line_up), (FRAME_WIDTH, line_up), (255, 0, 0), 2)  # Red line for up
#     cv2.line(frame, (0, line_down), (FRAME_WIDTH, line_down), (0, 0, 255), 2)  # Blue line for down
#     cv2.putText(frame, f"Up: {cnt_up}", (10, 40), font, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"Down: {cnt_down}", (10, 80), font, 1, (0, 255, 0), 2)

#     return frame


# # Process video frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process and display frame
#     processed_frame = process_frame(frame)
#     cv2.imshow("Vehicle Tracking with YOLOv8", processed_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
from ultralytics import YOLO

# Load YOLOv8 model pre-trained on COCO dataset
model = YOLO("yolov8n.pt")

# Initialize video capture
cap = cv2.VideoCapture("./Videos/video.mp4")

# Video frame properties
FRAME_WIDTH = 900
FRAME_HEIGHT = 500

# Line positions for counting
line_up = 250
line_down = 350

# Counters
cnt_up = 0
cnt_down = 0

# Font for text
font = cv2.FONT_HERSHEY_SIMPLEX

# Vehicle tracking dictionary
vehicle_tracks = {}  # Key: vehicle ID, Value: {'center': (cx, cy), 'counted_up': bool, 'counted_down': bool}
next_vehicle_id = 1  # Unique ID for vehicles


def process_frame(frame):
    global cnt_up, cnt_down, vehicle_tracks, next_vehicle_id

    # Resize frame for faster processing
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Perform YOLOv8 object detection
    results = model(frame, conf=0.4)  # Confidence threshold 0.4
    detections = results[0].boxes.data.cpu().numpy()  # Extract detections

    # Current frame vehicle centers
    current_frame_centers = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det  # Bounding box and class info
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Class IDs for vehicles in COCO (e.g., car, truck, bus, etc.)
        if int(cls) in [2, 3, 5, 7]:  # 2: car, 3: motorcycle, 5: bus, 7: truck
            # Calculate the center of the bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_frame_centers.append((cx, cy))

            # Match to existing vehicle or create a new one
            matched = False
            for vehicle_id, data in vehicle_tracks.items():
                old_cx, old_cy = data['center']
                if abs(cx - old_cx) < 50 and abs(cy - old_cy) < 50:  # Proximity threshold
                    vehicle_tracks[vehicle_id]['center'] = (cx, cy)
                    matched = True

                    # Upward crossing logic
                    if not data['counted_up'] and old_cy > line_up >= cy:
                        cnt_up += 1
                        vehicle_tracks[vehicle_id]['counted_up'] = True

                    # Downward crossing logic
                    if not data['counted_down'] and old_cy < line_down <= cy:
                        cnt_down += 1
                        vehicle_tracks[vehicle_id]['counted_down'] = True

            if not matched:
                # Assign a new vehicle ID
                vehicle_tracks[next_vehicle_id] = {
                    'center': (cx, cy),
                    'counted_up': False,
                    'counted_down': False,
                }
                next_vehicle_id += 1

            # Draw bounding box and vehicle ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{next_vehicle_id - 1}", (x1, y1 - 10), font, 0.5, (255, 255, 255), 1)

    # Remove lost vehicles (not detected in current frame)
    vehicle_tracks = {k: v for k, v in vehicle_tracks.items() if v['center'] in current_frame_centers}

    # Draw lines and display counters
    cv2.line(frame, (0, line_up), (FRAME_WIDTH, line_up), (255, 0, 0), 2)  # Red line for up
    cv2.line(frame, (0, line_down), (FRAME_WIDTH, line_down), (0, 0, 255), 2)  # Blue line for down
    cv2.putText(frame, f"Up: {cnt_up}", (10, 40), font, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Down: {cnt_down}", (10, 80), font, 1, (0, 255, 0), 2)

    return frame


# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process and display frame
    processed_frame = process_frame(frame)
    cv2.imshow("Vehicle Tracking with YOLOv8", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
