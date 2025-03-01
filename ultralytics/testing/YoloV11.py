import cv2
import os
from ultralytics import YOLO

def get_model_path():
    """Construct the absolute path for the model file."""
    # Define the base directory relative to the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Move two levels up to access the `Models` folder
    root_dir = os.path.abspath(os.path.join(base_dir, "..",))

    # Construct the path to the model
    model_path = os.path.join(root_dir, "models", "yolo11n.pt")

    return model_path

def load_model(model_path):
    """Load YOLO model from the given path."""
    return YOLO(model_path)
def process_frame(model, frame):
    """Process a frame with the YOLO model and return the processed frame."""
    results = model(frame)

    # Check if results are available and have detections
    if results:
        # Since results is a list, access the first element (assuming batch size is 1)
        results = results[0]
        # Plot the results (bounding boxes)
        processed_frame = results.plot()  # This will return the image with bounding boxes drawn
        return processed_frame
    else:
        return frame  # If no detections, return the original frame

def main():
    model_path = get_model_path()

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Load the model
    model = load_model(model_path)

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Process frame
        processed_frame = process_frame(model, frame)

        # Show processed frame
        cv2.imshow('YOLOv11 Object Detection', processed_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
