# Object-Detection-on-custom-dataset
This repository contains code and configurations for training a YOLOv8 object detector on a custom dataset. The project focuses on recognizing license plates using YOLOv8.

## Installation
### Clone the repository:

```
git clone https://github.com/JohnPaulPrabhu/Object-Detection-on-custom-dataset.git
cd Object-Detection-on-custom-dataset
```
### Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
### Configuration
Update the config.yaml file with the paths to your training and validation datasets. Ensure the dataset is in the correct format.

Training
To train the YOLOv8 model on your custom dataset, run:
```
python main.py
```
## Inference
To run inference on an image:
```
from ultralytics import YOLO

model = YOLO("runs/train/exp/weights/best.pt")
results = model.predict("path/to/your/image.jpg", save=True, imgsz=320, conf=0.5)
```
## Viewing Results
The inference results will be saved and can be viewed using:
```
from PIL import Image

for r in results:
    im_bgr = r.plot()
    im_rgb = Image.fromarray(im_bgr[..., ::-1])
    im_rgb.show()
    r.save(filename="results.jpg")
```
## Results
Training logs and weights are saved in the runs/train directory.
