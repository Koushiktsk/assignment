from ultralytics import YOLO
import cv2

# Load your custom YOLO model (fine-tuned for Player & Ball)
model_path = 'best.pt'  # your custom model
model = YOLO(model_path)

# Load video
video_path = '15sec_input_720p.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_player_ball_detect.avi', fourcc, fps, (width, height))

# Class names based on training (update this list according to your model's classes)
class_names = ['player', 'ball']  # <- Adjust if your model has more classes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = result

        # Handle unknown classes safely
        if int(cls) < len(class_names):
            label = class_names[int(cls)]
        else:
            label = f"class_{int(cls)}"  # Fallback if class is unknown

        if score > 0.4:  # Confidence threshold
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    out.write(frame)
    cv2.imshow('Player and Ball Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()