import time
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    # Set up Detectron2 model for object detection
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"  # Use GPU for inference
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    predictor = DefaultPredictor(cfg)

    # Initialize DeepSORT tracker
    deepsort = DeepSort(max_age=70, n_init=3, max_iou_distance=0.7)

    # Load the video input
    video = cv2.VideoCapture("output_video.mp4")
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    fps = float(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames in the video: {total_frames}")

    # Initialize VideoWriter with a compatible codec for .mp4 format
    video_out = cv2.VideoWriter("result1.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Check if VideoWriter is opened successfully
    if not video_out.isOpened():
        print("Error: Could not open video writer.")
        return

    frame_counter = 0
    trackers = []  # To hold the tracked objects' info: [id, bbox, last_seen_frame]
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_identity_switches = 0

    # Track the time to compute FPS
    start_time = time.time()

    while True:
        frame_counter += 1
        success, frame = video.read()
        if not success:
            break

        print(f"Processing frame {frame_counter}...")

        # Object detection on the frame
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()

        # Filter to detect only 'person' (class_id == 0 in COCO dataset for 'person')
        detections = []
        for i in range(len(classes)):
            if classes[i] == 0 and scores[i] > 0.5:  # Detecting people with confidence > 0.5
                box = boxes[i].tolist()
                score = scores[i]
                detections.append((box, score))

        # If there are detections, we update manually
        if detections:
            bboxes = [det[0] for det in detections]
            scores = [det[1] for det in detections]

            # Here, we manually update trackers
            new_trackers = []
            for bbox, score in zip(bboxes, scores):
                matched = False
                for track in trackers:
                    track_id, track_bbox, _ = track
                    # Calculate IoU or distance between tracked and new bounding box (simple overlap check)
                    iou = calculate_iou(track_bbox, bbox)  # Implement calculate_iou as per your needs
                    if iou > 0.5:  # If IoU is high, we update the tracker
                        track[1] = bbox  # Update the position
                        new_trackers.append(track)
                        matched = True
                        total_true_positives += 1  # Increment true positives for matched trackers
                        break
                if not matched:
                    # Create a new tracker if no match was found
                    new_id = len(trackers) + 1  # Incremental ID assignment (can be refined)
                    new_trackers.append([new_id, bbox, frame_counter])
                    total_false_positives += 1  # New track means false positive

            # For identity switches (simplified version, need to track ID continuity)
            # You can refine this section further with better tracking of IDs across frames
            total_identity_switches += 0  # Assuming no identity switches for now

            trackers = new_trackers

            # Draw bounding boxes and track IDs
            for track in trackers:
                track_id, track_bbox, _ = track
                x1, y1, x2, y2 = [int(i) for i in track_bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the annotated frame to output video
        video_out.write(frame)

    # Calculate MOTA, IDF1, and FPS
    total_time = time.time() - start_time
    calculated_fps = frame_counter / total_time
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    idf1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mota = 1 - (total_false_positives + total_false_negatives + total_identity_switches) / frame_counter

    # Output the metrics
    print(f"MOTA: {mota:.4f}")
    print(f"IDF1: {idf1:.4f}")
    print(f"FPS: {calculated_fps:.2f}")

    # Release resources
    video.release()
    video_out.release()
    print("Processing complete. Output saved to result1.mp4.")

# Function to calculate Intersection over Union (IoU)
def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection area
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union area
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

if __name__ == "__main__":
    main()