import cv2
from ultralytics import YOLO
import argparse

def detect_class_in_video(video_path, output_path, model, save_txt=False, txt_path='results.txt', conf=0.25, iou=0.7, allowed_classes=None):
    # Support webcam indices passed as strings
    if isinstance(video_path, str) and video_path.isdigit():
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open source {video_path}")
        return

    # Defer writer init until first valid frame; set fps fallback
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps_val) if fps_val and fps_val > 0 else 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    f = open(txt_path, 'w') if save_txt else None
    # Robust class names with fallback matching README_H.md
    names_attr = getattr(model, 'names', getattr(model.model, 'names', None))
    if isinstance(names_attr, (list, tuple)):
        names = {i: n for i, n in enumerate(names_attr)}
    elif isinstance(names_attr, dict):
        names = names_attr
    else:
        names = {0: 'non_violence', 1: 'violence'}

    counts = {}
    total_frames = 0
    violence_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        total_frames += 1

        results = model(frame, conf=conf, iou=iou)
        for detection in results[0].boxes:
            class_id = int(detection.cls)
            if allowed_classes is not None and class_id not in allowed_classes:
                continue
            confidence = float(detection.conf)

            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = names[class_id] if class_id in names else f'Class {class_id}'
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if f is not None:
                name_part = label
                f.write(f'{int(class_id)} {name_part} {confidence:.6f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n')
            counts[class_id] = counts.get(class_id, 0) + 1
            if class_id == 1:
                violence_frames +=1

        if out is not None:
            out.write(frame)

    cap.release()
    if out is not None:
        out.release()
    if f is not None:
        # Summary
        f.write('--- summary ---\n')
        if counts:
            for cid, c in sorted(counts.items()):
                label = names[cid] if cid in names else f'Class {cid}'
                f.write(f'{cid} {label}: {c}\n')
        if total_frames:
            ratio = violence_frames / total_frames
            target_label = names.get(1, f'Class {1}')
            f.write(f'frames_total: {total_frames}\n')
            f.write(f'frames_with_{target_label}: {violence_frames}\n')
            f.write(f'{target_label}_frame_ratio: {ratio:.4f}\n')
        f.close()
    print(f"Processed video saved at {output_path}")

def detect_class_in_image(image_path, output_path, model, save_txt=False, txt_path='results.txt', conf=0.25, iou=0.7, allowed_classes=None):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: cannot read image {image_path}")
        return
    results = model(frame, conf=conf, iou=iou)

    f = open(txt_path, 'w') if save_txt else None
    # Robust class names with fallback matching README_H.md
    names_attr = getattr(model, 'names', getattr(model.model, 'names', None))
    if isinstance(names_attr, (list, tuple)):
        names = {i: n for i, n in enumerate(names_attr)}
    elif isinstance(names_attr, dict):
        names = names_attr
    else:
        names = {0: 'non_violence', 1: 'violence'}

    counts = {}
    for detection in results[0].boxes:
        class_id = int(detection.cls)
        if allowed_classes is not None and class_id not in allowed_classes:
            continue
        confidence = float(detection.conf)

        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = names[class_id] if class_id in names else f'Class {class_id}'
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if f is not None:
            name_part = label
            f.write(f'{int(class_id)} {name_part} {confidence:.6f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n')
        counts[class_id] = counts.get(class_id, 0) + 1

    if f is not None:
        # Summary
        if counts:
            f.write('--- summary ---\n')
            for cid, c in sorted(counts.items()):
                label = names[cid] if cid in names else f'Class {cid}'
                f.write(f'{cid} {label}: {c}\n')
        f.close()
    cv2.imwrite(output_path, frame)
    print(f"Processed image saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 detection script.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the YOLOv8 model weights.')
    parser.add_argument('--source', type=str, required=True, help='Path to the input video or image.')
    parser.add_argument('--save-txt', action='store_true', help='Save detection results to a text file.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections.')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS.')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --classes 0, or --classes 0 2 3')
    args = parser.parse_args()

    model = YOLO(args.weights)
    
    output_path = "output." + args.source.split('.')[-1]

    if args.source.lower().endswith(('.png', '.jpg', '.jpeg')):
        detect_class_in_image(
            args.source,
            output_path,
            model,
            save_txt=args.save_txt,
            txt_path='results.txt',
            conf=args.conf,
            iou=args.iou,
            allowed_classes=args.classes
        )
    else:
        detect_class_in_video(
            args.source,
            output_path,
            model,
            save_txt=args.save_txt,
            txt_path='results.txt',
            conf=args.conf,
            iou=args.iou,
            allowed_classes=args.classes
        )

    if args.save_txt:
        print("Detection results saved to results.txt")


if __name__ == "__main__":
    main()

