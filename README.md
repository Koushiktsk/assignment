## assignment : Re-Identification in a single Feed

# How to Run
1) Make sure Yolo and OpenCV downloaded in the machine.
2) download the model(Ultralytics YOLOv11) :- https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view 
3) download the app.py file
4) download the video file 
5) make sure all the the model, video file and app.py file are in same directory

# Methodologies

Core Methodologies:
  YOLOv8 Object Detection:
    Uses your custom-trained best.pt mode
    Processes each frame to detect players and balls
    Outputs bounding boxes with class labels and confidence scores
  Frame Processing Pipeline:
    Video input → Frame-by-frame processing → Output video
    Uses OpenCV for video capture and display
    Maintains original video resolution and FPS
  Detection Visualization:
    Draws green bounding boxes around detected objects
    Labels each detection with:
    Class name ('player' or 'ball')
    Confidence score (formatted to 2 decimal places)
    Only shows detections above 0.4 confidence threshold
  Safety Mechanisms:
    Handles unknown classes with fallback labeling (class_X)
    Checks class index bounds to prevent crashes
    Properly releases resources when finished




