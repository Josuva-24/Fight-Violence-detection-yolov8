# Violence Detection with YOLOv8 - Tutorial

This project implements a violence detection system using YOLOv8 object detection model. The model can identify potential violent acts in real-time video streams or images.

## Table of Contents
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Training](#model-training)
4. [Inference](#inference)
5. [Project Structure](#project-structure)
6. [Technologies Used](#technologies-used)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Fight-Violence-detection-yolov8.git
   cd Fight-Violence-detection-yolov8
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

The dataset preparation process is handled by `prepare_data.py`:
- It organizes your video/image dataset into the required format
- Splits data into train/validation/test sets
- Creates the YAML configuration file required by YOLO

Run the dataset preparation:
```bash
python prepare_data.py
```

## Model Training

To train the YOLOv8 model:
```bash
# Train the model with default settings
python train.py

# Or customize training parameters
python train.py --img 640 --batch 16 --epochs 100 --data violence_data.yaml
```

## Inference

To run inference on video/images:
```bash
# Detect violence in a video
python detect.py --source path/to/video.mp4 --weights yolo11n.pt

# Detect violence in images
python detect.py --source path/to/images/ --weights yolo11n.pt

# Real-time detection from webcam
python detect.py --source 0 --weights yolo11n.pt
```

## Project Structure

```
Fight-Violence-detection-yolov8/
├── detect.py          # Main detection script
├── prepare_data.py    # Data preparation script
├── violence_data.yaml # Dataset configuration
├── requirements.txt   # Dependencies
├── README_HOW.md      # This file
├── README_H.md        # Main README
├── Yolo_nano_weights.pt    # Pre-trained nano model weights
├── yolo_small_weights.pt   # Pre-trained small model weights
├── yolo11n.pt              # YOLOv11 nano weights
├── testing_code.ipynb      # Jupyter notebook for testing
└── runs/            # Training outputs and logs
```

## Technologies Used

- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- NumPy
- Matplotlib
- Pandas

## Model Architecture

This project uses YOLO (You Only Look Once) architecture for real-time object detection:
- YOLOv8 nano: Lightweight model for faster inference
- YOLOv8 small: Balanced model for accuracy and speed
- YOLOv11: Latest YOLO model with improved performance

## Results

The model detects potential violent acts by:
1. Identifying objects in video frames
2. Analyzing object interactions and movements
3. Flagging specific patterns associated with violence
4. Providing real-time alerts when violence is detected

## Performance Metrics

- Detection accuracy: [To be filled based on your evaluation]
- Inference speed: [To be filled based on your testing]
- Precision/Recall: [To be filled based on your evaluation]

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/your-username/Fight-Violence-detection-yolov8](https://github.com/your-username/Fight-Violence-detection-yolov8)