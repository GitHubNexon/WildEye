# train.py
import os
from ultralytics import YOLO

# Variable for dataset name
dataset_name = "WildEye"

# Updated path to the YAML file (the dataset configuration file)
data_yaml = r"C:\VSCode_Projects\DataSets\WildEye1"

# Ensure that the dataset is available
if not os.path.exists(data_yaml):
    raise FileNotFoundError(f"Dataset YAML file not found at {data_yaml}")

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

# Path to save the trained model and results
save_path = r"C:\VSCode_Projects\WildEye\ultralytics\runs"

# Train the model
model.train(
    data=data_yaml,         # Path to the dataset YAML file
    # epochs=50,            # Number of epochs to train
    epochs=10,              # Number of epochs to train --testing
    # imgsz=640,            # Image size for training
    imgsz=416,              # Image size for training --testing
    batch=16,             # Batch size (adjust based on your system's memory)
    batch=8,                # Batch size (adjust based on your system's memory) --testing
    project=save_path,      # Custom path for saving results
    augment=True,
    weight_decay=0.0005,    # Regularization (to prevent overfitting)
    warmup_epochs=3,        # Warmup epochs (helps stabilize early training)
    name='WildEye',          # Name for the saved model

    patience=3,             # If no improvement in validation mAP after 3 epochs, stop training
    # save_period=3,        # Save the model every 2 epochs (you can adjust this)
    device='0',           # Explicitly use GPU (if available)
    # device='cpu'          # using the CPU 
)

print(f"Training completed for {dataset_name}!")
