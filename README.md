# Fight/Violence Detection Dataset and Model Overview

This repository contains a dataset and YOLOv8 models (nano and small) trained to detect fights/violence and non-violence/no-fight in both videos and images. The models are optimized for surveillance and security applications where detecting physical confrontations is crucial.

## Dataset Overview

- **Dataset Classes:** The dataset consists of two classes:
  1. **Violence/Fight:** Instances where physical violence is present.
  2. **NoViolence/NoFight:** Instances with no physical confrontations.
  
- **Data Format:** 
  - **Videos** and **Images** are labeled accordingly for each class.
  - The dataset is designed for training deep learning models like YOLOv8 for violence detection.

## Models

- **YOLOv8 Models:** 
  - We have primarily trained **YOLOv8-nano** and **YOLOv8-small** models.
  - These models are lightweight and efficient, making them suitable for real-time detection tasks in resource-constrained environments.

## Purpose

The models are trained to accurately detect violent events in various settings, including crowds, public spaces, and sports activities.

### Key Features:
- **Single Class Detection:**
  - The attached code is specifically designed to detect **one class** at a time, with the focus being on the **Violence/Fight** class.
  - If the purpose is to detect only **Violence/Fight**, the models and code are pre-configured for this task.
  - Non-violence events are ignored during detection, allowing the model to concentrate solely on identifying violent actions.

## Code and Usage Instructions

### Pre-requisites:
- Python 3.8 or higher
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV

### Running the Detection:

1. **Clone the Repository**:
    ```bash
    git clone (https://github.com/Musawer1214/Fight-Violence-detection-yolov8)
    cd <repository-directory>
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Single Class Detection (Violence/Fight)**:
   The provided script detects only the **Violence/Fight** class in both videos and images.

    ```bash
    python detect.py --weights best.pt --source <input-video-or-image-path> --class 1 --save-txt
    ```

   Replace `<input-video-or-image-path>` with the path to the video or image you wish to analyze. The **class ID** for **Violence/Fight** is **1**.

4. **Model Weights**:
    - The `best.pt` file contains the pre-trained YOLOv8-nano or YOLOv8-small model optimized for detecting violence/fights.

## Notes

- **Model Performance:** The models are trained on a diverse set of images to generalize across different environments. However, additional fine-tuning may be required depending on your specific use case.
- **Future Enhancements:** We plan to extend the dataset and include more diverse scenarios to improve detection accuracy, including sports, public gatherings, and more.

---

## License

This project is licensed under the [MIT License](LICENSE).

