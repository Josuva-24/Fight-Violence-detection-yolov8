
import os
import cv2
import random
import shutil

def extract_frames(video_path, output_dir, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open source {video_path}")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_dir, f"{os.path.basename(video_path)}_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        label_filename = os.path.join(output_dir.replace("images", "labels"), f"{os.path.basename(video_path)}_{frame_count}.txt")
        with open(label_filename, "w") as f:
            f.write(f"{label} 0.5 0.5 1 1\n") # Assuming the whole frame is the object
            
        frame_count += 1
        
    cap.release()

def create_dataset(dataset_path, output_path):
    images_train_path = os.path.join(output_path, "images", "train")
    images_val_path = os.path.join(output_path, "images", "val")
    labels_train_path = os.path.join(output_path, "labels", "train")
    labels_val_path = os.path.join(output_path, "labels", "val")
    
    os.makedirs(images_train_path, exist_ok=True)
    os.makedirs(images_val_path, exist_ok=True)
    os.makedirs(labels_train_path, exist_ok=True)
    os.makedirs(labels_val_path, exist_ok=True)
    
    non_violent_path = os.path.join(dataset_path, "non-violent")
    violent_path = os.path.join(dataset_path, "violent")
    
    video_files = []
    for cam_dir in os.listdir(non_violent_path):
        for video in os.listdir(os.path.join(non_violent_path, cam_dir)):
            video_files.append((os.path.join(non_violent_path, cam_dir, video), 0))
            
    for cam_dir in os.listdir(violent_path):
        for video in os.listdir(os.path.join(violent_path, cam_dir)):
            video_files.append((os.path.join(violent_path, cam_dir, video), 1))
            
    random.shuffle(video_files)
    
    split_ratio = 0.8
    split_index = int(len(video_files) * split_ratio)
    
    train_files = video_files[:split_index]
    val_files = video_files[split_index:]
    
    for video_path, label in train_files:
        extract_frames(video_path, images_train_path, label)
        
    for video_path, label in val_files:
        extract_frames(video_path, images_val_path, label)

if __name__ == "__main__":
    dataset_path = "/mnt/disk1/Hackathon/Fight-Violence-detection-yolov8/A-Dataset-for-Automatic-Violence-Detection-in-Videos/violence-detection-dataset"
    output_path = "/mnt/disk1/Hackathon/Fight-Violence-detection-yolov8/violence_dataset"
    create_dataset(dataset_path, output_path)
