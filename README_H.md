# Fight/Violence Detection Model - What It Can Do

## Overview
This repository contains a specialized YOLOv8-based deep learning model trained to detect and classify violent/fighting behaviors in real-time video streams and static images. The model is designed for security, surveillance, and safety applications.

## Model Capabilities

### 1. Core Detection Functions
- **Violence Detection**: Identifies physical fights, aggressive behavior, and violent confrontations
- **Non-Violence Classification**: Distinguishes between violent and non-violent activities
- **Real-time Processing**: Performs inference at 8-12ms per frame for real-time applications
- **Multi-scale Detection**: Detects violence at various distances and scales within the frame

### 2. Input Types Supported
- **Video Files**: MP4, AVI, MOV, and other common video formats
- **Live Video Streams**: Real-time camera feeds and surveillance streams
- **Static Images**: JPG, PNG, and other image formats
- **Batch Processing**: Multiple files can be processed sequentially

### 3. Model Variants Available
- **YOLOv8-Nano**: Ultra-lightweight model (6.25MB) for edge devices and mobile applications
- **YOLOv8-Small**: Balanced model (22.5MB) offering better accuracy for standard applications

### 4. Detection Classes
The model is trained to identify two primary classes:
1. **Violence/Fight** (Class ID: 1): Physical altercations, aggressive behavior, fighting
2. **Non-Violence/No-Fight** (Class ID: 0): Normal activities, peaceful interactions

### 5. Performance Characteristics
- **Speed**: 8-18ms inference time per frame on GPU
- **Accuracy**: Optimized for real-world surveillance scenarios
- **Efficiency**: Lightweight enough for deployment on edge devices
- **Robustness**: Trained on diverse scenarios including crowds, sports, and public spaces

### 6. Output Capabilities
- **Bounding Box Detection**: Precise localization of violent activities
- **Confidence Scores**: Probability ratings for each detection
- **Visual Annotations**: Automatic labeling with colored boxes and confidence text
- **Text Logging**: Detection results saved to text files with coordinates and confidence
- **Processed Video/Image Output**: Annotated media files with detected violence highlighted

### 7. Technical Specifications
- **Framework**: YOLOv8 (Ultralytics)
- **Input Resolution**: 384x640 pixels (optimized)
- **Inference Backend**: PyTorch
- **Hardware Requirements**: 
  - GPU: CUDA-compatible (recommended)
  - CPU: Multi-core processor (minimum)
  - RAM: 4GB+ recommended
  - Storage: 100MB+ for model weights

### 8. Use Cases
- **Security Surveillance**: Automatic alert systems for security personnel
- **Public Safety**: Monitoring crowded areas, events, and public spaces
- **Educational Institutions**: Campus safety and anti-bullying systems
- **Sports Analysis**: Detecting aggressive behavior in competitive sports
- **Healthcare**: Monitoring patient behavior in medical facilities
- **Transportation**: Safety monitoring in buses, trains, and stations

### 9. Integration Capabilities
- **Command Line Interface**: Simple execution via terminal commands
- **Python API**: Direct integration into existing applications
- **Real-time Streaming**: Compatible with live video feeds
- **Batch Processing**: Automated processing of multiple files
- **Custom Configuration**: Adjustable confidence thresholds and detection parameters

### 10. Deployment Options
- **Local Processing**: On-premises deployment for privacy-sensitive applications
- **Edge Computing**: Deployment on IoT devices and edge servers
- **Cloud Integration**: Scalable cloud-based processing
- **Mobile Applications**: Integration into mobile security apps
- **Embedded Systems**: Deployment on surveillance cameras and security devices

### 11. Customization Features
- **Selective Class Detection**: Focus on specific violence types (Class 1 only)
- **Confidence Thresholds**: Adjustable sensitivity settings
- **Output Formats**: Multiple output options (video, image, text logs)
- **Region of Interest**: Focus detection on specific areas of the frame
- **Alert Systems**: Integration with notification and alarm systems

### 12. Real-world Applications
- **Smart Cities**: Automated violence detection in urban surveillance networks
- **Event Security**: Crowd monitoring at concerts, sports events, and gatherings
- **Retail Security**: Loss prevention and customer safety in stores
- **Transportation Security**: Safety monitoring in airports, stations, and vehicles
- **Healthcare Monitoring**: Patient safety in hospitals and care facilities
- **School Safety**: Anti-bullying and violence prevention in educational settings

### 13. Model Limitations and Considerations
- **Single Focus**: Optimized specifically for violence detection (not general object detection)
- **Context Dependency**: May require human verification for complex scenarios
- **Lighting Conditions**: Performance may vary under poor lighting
- **Camera Angle**: Best performance with clear, unobstructed views
- **Privacy**: Ensure compliance with local privacy laws and regulations

### 14. Performance Metrics
- **Inference Speed**: 8-18ms per frame
- **Model Size**: 6.25MB (nano) / 22.5MB (small)
- **Input Processing**: 2-3ms preprocessing time
- **Output Generation**: 1-2ms postprocessing time
- **Memory Usage**: Low memory footprint suitable for edge deployment

This model represents a specialized solution for violence detection that balances accuracy, speed, and resource efficiency, making it suitable for a wide range of real-world security and safety applications.


## Usage

Install deps:

    pip install -r requirements.txt

Run detection (choose one of the included weights):

    python detect.py --weights yolo_small_weights.pt --source <input-video-or-image-path> --class 1 --save-txt
    # or
    python detect.py --weights Yolo_nano_weights.pt --source <input-video-or-image-path> --class 1 --save-txt

Notes:
- Class 1 = violence/fight, Class 0 = non_violence.
- results.txt will include a summary and per-frame stats for videos.
