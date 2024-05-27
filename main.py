from ultralytics import YOLO  # Importing the YOLO class from ultralytics library
from PIL import Image  # Importing the Image class from PIL library for image handling

def main():
    # Load the model configuration and build the model from scratch
    model = YOLO("yolov8n-obb.yaml")
    print("Starting training...")

    # Training the model with specified parameters
    results = model.train(
        data="config.yaml",  # Path to your data configuration file
        epochs=30,           # Number of training epochs
        device=0,            # Specify the device (0 for GPU, 'cpu' for CPU)
        batch=16,            # Adjust batch size based on your GPU memory
        lr0=0.001,           # Initial learning rate
        weight_decay=0.0005, # Weight decay for regularization
        save_period=1,       # Save model every epoch
        name='exp',          # Experiment name, results will be saved in 'runs/train/exp'
        project='runs/train' # Project directory
    )
    print("Training completed successfully.")
    
    # Load a model for validation and inference
    model = YOLO("D:/Code/Object-Detector-on-custom-dataset/runs/train/exp/weights/best.pt")  # Load a custom model
    
    # Validate the model on validation dataset
    metrics = model.val()  # No arguments needed, dataset and settings remembered
    metrics.box.map  # Mean Average Precision (mAP) for IoU=0.5:0.95
    metrics.box.map50  # mAP for IoU=0.5
    metrics.box.map75  # mAP for IoU=0.75
    metrics.box.maps  # A list containing mAP for IoU=0.5:0.95 for each category

    # Load the custom trained model again
    model = YOLO("path/to/best.pt")
    
    # Run inference on an image
    results = model("path/to/img")  # Inference results list
    
    # Iterate over results and process them
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # Get the BGR-order numpy array of the results
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # Convert to RGB-order PIL image
    
        # Show results to screen (in supported environments)
        r.show()
    
        # Save results to disk with filenames based on index
        r.save(filename=f"results{i}.jpg")
    
if __name__ == '__main__':
    main()  # Call the main function to start the process