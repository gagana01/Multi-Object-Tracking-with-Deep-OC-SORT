import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog


# Function to print tracking metrics (frame number, object ID, bounding box coordinates, and confidence)
def print_metrics(frame_number, obj_id, bbox, confidence):
    bb_left, bb_top, bb_width, bb_height = bbox
    print(f"{frame_number},{obj_id},{bb_left},{bb_top},{bb_width},{bb_height},{confidence},-1,-1,-1")


# Function to associate predictions with existing objects based on IOU (Intersection over Union)
def associate_predictions(existing_objects, predictions, iou_threshold=0.9):
    matched_objects = {}
    # Loop through all predictions and compare with existing objects to find matches
    for pred_id, (pred_class, pred_box) in enumerate(predictions):
        max_iou = 0
        matched_id = None
        for object_id, (obj_class, obj_box) in existing_objects.items():
            if obj_class == pred_class:  # Match classes before comparing boxes
                iou = calculate_iou(obj_box, pred_box)  # Calculate IOU for matching
                if iou > max_iou:
                    max_iou = iou
                    matched_id = object_id
        if matched_id is not None and max_iou > iou_threshold:
            matched_objects[pred_id] = matched_id  # Track matched object
        else:
            matched_objects[pred_id] = None  # No match found
    return matched_objects


# Function to update existing objects based on matched predictions
def update_existing_objects(existing_objects, predictions, matched_objects):
    for pred_id, (pred_class, pred_box) in enumerate(predictions):
        matched_id = matched_objects[pred_id]
        if matched_id is not None:
            existing_objects[matched_id] = (pred_class, pred_box)  # Update matched object
        else:
            new_id = max(existing_objects.keys()) + 1 if existing_objects else 0
            existing_objects[new_id] = (pred_class, pred_box)  # Assign new ID for new object
    return existing_objects


# Function to calculate IOU (Intersection over Union) between two bounding boxes
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_g, y1_g, x2_g, y2_g = box2
    x_intersection = max(0, min(x2, x2_g) - max(x1, x1_g))
    y_intersection = max(0, min(y2, y2_g) - max(y1, y1_g))
    intersection_area = x_intersection * y_intersection
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_g - x1_g) * (y2_g - y1_g)
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0


# Function to draw bounding boxes with IDs and print tracking metrics for each frame
def draw_boxes_with_ids(frame_counter, frame, instances, existing_objects):
    predictions = instances.to("cpu")  # Move predictions to CPU
    matched_objects = associate_predictions(
        existing_objects,
        zip(predictions.pred_classes, predictions.pred_boxes.tensor.cpu().numpy())  # Associate predictions with existing objects
    )
    updated_objects = update_existing_objects(
        existing_objects,
        zip(predictions.pred_classes, predictions.pred_boxes.tensor.cpu().numpy()),
        matched_objects
    )
    
    # Loop through each prediction and draw the bounding box along with the object ID
    for pred_id, (pred_class, pred_box) in enumerate(
        zip(predictions.pred_classes, predictions.pred_boxes.tensor.cpu().numpy())
    ):
        object_id = matched_objects.get(pred_id)  # Get the matched object ID
        if object_id is not None:
            if pred_class.item() != 0:  # Skip non-person classes
                continue
            draw_text = f"ID: {object_id}, Class: {pred_class.item()}"  # Text to display on frame
            text_position = (int(pred_box[0]), int(pred_box[1] - 10))  # Position for text
            cv2.putText(frame, draw_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            bb_width = int(pred_box[2] - pred_box[0])
            bb_height = int(pred_box[3] - pred_box[1])
            print_metrics(frame_counter, object_id, (int(pred_box[0]), int(pred_box[1]), bb_width, bb_height), 1.0)  # Print metrics for the detected object
    return frame


# Main function to run the tracking system
if __name__ == "__main__":
    # Configuration for Detectron2 Mask R-CNN model
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # Load config for Mask R-CNN
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set threshold for detection
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Load pre-trained weights

    existing_objects = {}  # Dictionary to store existing objects with their IDs
    predictor = DefaultPredictor(cfg)  # Initialize predictor with the configuration

    # Open video file for reading and prepare for output
    video = cv2.VideoCapture("/content/output_video.mp4")
    fps = float(video.get(cv2.CAP_PROP_FPS))  # Get FPS from input video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get frame width
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get frame height
    video_out = cv2.VideoWriter("/content/result_maskRcnn.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))  # Set up video writer for output

    frame_counter = 0  # Initialize frame counter

    # Process each frame in the video
    while True:
        success, frame = video.read()
        if not success:  # Break if video ends
            break
        frame_counter += 1  # Increment frame counter
        outputs = predictor(frame)  # Get predictions from the Mask R-CNN model
        instances = outputs["instances"]  # Get the instances (detections)
        person_instances = instances[instances.pred_classes == 0]  # Filter for persons only (class 0)
        annotated_frame = draw_boxes_with_ids(frame_counter, frame, person_instances, existing_objects)  # Draw boxes with IDs
        video_out.write(annotated_frame)  # Write annotated frame to output video

    print("Processing complete.")
    video_out.release()  # Release the output video writer
    video.release()  # Release the input video
