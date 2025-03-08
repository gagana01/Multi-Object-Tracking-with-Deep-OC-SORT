from google.colab.patches import cv2_imshow
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import numpy as np

# Function to select keypoints from the detected keypoints
def select_keypoints(instances, frame):
    # Select a subset of keypoints (e.g., head, shoulders, wrists)
    selected_keypoints = [5, 6, 7, 8, 11, 12, 13, 14]
    keypoints_hsv = []

    # Loop through detected keypoints and select the ones from the subset
    for keypoints in instances.pred_keypoints:
        hsv_per_keypoint = []
        for idx in selected_keypoints:
            x, y, _ = keypoints[idx]
            # Check if keypoint coordinates are valid within the frame
            if 0 <= int(x) < frame.shape[1] and 0 <= int(y) < frame.shape[0]:
                # Convert BGR values of keypoint to HSV
                bgr_color = frame[int(y), int(x)].astype(np.uint8).reshape(1, 1, 3)
                hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
                hsv_per_keypoint.extend(hsv_color)  # Add HSV color values
                hsv_per_keypoint.append(x)  # Add x-coordinate
                hsv_per_keypoint.append(y)  # Add y-coordinate
        keypoints_hsv.append(hsv_per_keypoint)
    return keypoints_hsv


# Function to track objects based on the proximity of keypoints in HSV and XY space
def track_objects(tracking_objects, keypoints_hsv):
    updated_objects = {}
    # Loop through previously tracked objects
    for obj_id, hsv2 in tracking_objects.items():
        object_found = False
        # Compare each tracked object with the new detected keypoints
        for hsv1 in keypoints_hsv:
            # Calculate differences in HSV values and XY coordinates
            diff_hsv = sum(abs(int(hsv2[i]) - int(hsv1[i])) for i in range(3))
            diff_xy = abs(hsv2[-2] - hsv1[-2]) + abs(hsv2[-1] - hsv1[-1])
            if diff_hsv <= 960 and diff_xy <= 240:  # Threshold for matching object
                updated_objects[obj_id] = hsv1  # Object matched, update tracking
                object_found = True
                keypoints_hsv.remove(hsv1)  # Remove matched keypoint
                break
        # If no match found, keep the previous object state
        if not object_found:
            updated_objects[obj_id] = hsv2
    return updated_objects, keypoints_hsv


# Function to assign new object IDs for any newly detected objects
def assign_object_ids(tracking_objects, keypoints_hsv, tracking_id):
    # Assign new unique IDs for objects not previously tracked
    for hsv in keypoints_hsv:
        tracking_objects[tracking_id] = hsv
        tracking_id += 1  # Increment ID for next new object
    return tracking_objects, tracking_id


# Main code execution
if __name__ == "__main__":
    # Load the keypoint detection model configuration
    cfg_keypoint = get_cfg()
    cfg_keypoint.MODEL.DEVICE = "cpu"  # Use CPU for inference
    cfg_keypoint.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg_keypoint.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for object detection
    cfg_keypoint.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

    # Initialize the predictor
    predictor = DefaultPredictor(cfg_keypoint)
    
    # Open the input video for processing
    video = cv2.VideoCapture("/content/output_video.mp4")  # Replace with your input video path
    fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video width
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video height
    video_out = cv2.VideoWriter("/content/keypoints_result.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    tracking_objects = {}  # Dictionary to store currently tracked objects
    tracking_id = 0  # Initial tracking ID for new objects
    frame_number = 0  # Frame counter for the video

    while True:
        success, frame = video.read()  # Read a new frame from the video
        if not success:
            break  # Stop when the video ends

        frame_number += 1  # Increment frame number
        outputs = predictor(frame)  # Run keypoint prediction on the frame
        instances = outputs["instances"].to("cpu")  # Get instances of detected keypoints

        # If keypoints are detected, process them
        if instances.has("pred_keypoints"):
            keypoints_hsv = select_keypoints(instances, frame)  # Extract keypoints' HSV and positions
            tracking_objects, keypoints_hsv = track_objects(tracking_objects, keypoints_hsv)  # Track objects based on proximity
            tracking_objects, tracking_id = assign_object_ids(tracking_objects, keypoints_hsv, tracking_id)  # Assign new IDs to new objects

            # Display object IDs and keypoints on the frame
            for obj_id, pt in tracking_objects.items():
                if len(pt) >= 2:  # Ensure keypoints have sufficient data
                    x, y = int(pt[-2]), int(pt[-1])  # Extract x, y coordinates
                    cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Display object ID
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a circle around the keypoint

        # Visualize the predicted keypoints on the frame
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg_keypoint.DATASETS.TRAIN[0]))
        out_frame = v.draw_instance_predictions(instances).get_image()[:, :, ::-1]

        video_out.write(out_frame)  # Write the processed frame to the output video
        cv2_imshow(out_frame)  # Display the frame in Colab
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Allow quitting by pressing 'q'
            break

    video.release()  # Release the video object
    video_out.release()  # Release the output video object
    cv2.destroyAllWindows()  # Close any OpenCV windows
    print("Processing Complete!")  # Print completion message
