import cv2
import pytesseract
from pytesseract import Output
import os
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

def extract_track_info(frame, frame_counter):
    """
    Extracts tracking information from a frame using OCR and draws bounding boxes with IDs.
    """
    # Convert frame to grayscale for better OCR accuracy
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess frame with adaptive thresholding
    processed_frame = cv2.adaptiveThreshold(
        gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # OCR to extract text from frame
    ocr_result = pytesseract.image_to_data(
        processed_frame,
        config="--psm 6",  # Assume a single block of text
        output_type=Output.DICT,
    )

    track_ids = []
    h, w, _ = frame.shape

    # Iterate through OCR results and draw bounding boxes around detected IDs
    for i in range(len(ocr_result["text"])):
        if ocr_result["text"][i].strip().isdigit():
            x, y, w, h = (
                ocr_result["left"][i],
                ocr_result["top"][i],
                ocr_result["width"][i],
                ocr_result["height"][i],
            )
            track_id = int(ocr_result["text"][i].strip())
            track_ids.append(track_id)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return frame, track_ids

def plot_metrics(track_history):
    """
    Plots metrics including track counts over time and lifespan distributions.
    """
    # Lifespan distribution
    lifespans = [max(frames) - min(frames) + 1 for frames in track_history.values()]
    track_ids = list(track_history.keys())

    # Plot 1: Lifespan distribution
    plt.figure(figsize=(10, 5))
    plt.bar(track_ids, lifespans, color="skyblue")
    plt.title("Track Lifespan Distribution")
    plt.xlabel("Track ID")
    plt.ylabel("Lifespan (frames)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Plot 2: Track counts over time
    time_track_counts = {}
    for track_frames in track_history.values():
        for frame in track_frames:
            time_track_counts[frame] = time_track_counts.get(frame, 0) + 1

    frames = sorted(time_track_counts.keys())
    counts = [time_track_counts[frame] for frame in frames]

    plt.figure(figsize=(10, 5))
    plt.plot(frames, counts, marker="o", color="orange")
    plt.title("Number of Tracks Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Track Count")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

def main():
    # Path to the input video
    video_path = "/content/result1 (2).mp4"
    if not os.path.exists(video_path):
        print("Error: Input video not found!")
        return

    # Load the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open the video!")
        return

    # Initialize variables
    frame_counter = 0
    frame_skip = 5  # Process every 5th frame to reduce workload
    track_history = {}

    while True:
        success, frame = video.read()
        if not success:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        # Resize frame to reduce memory usage
        frame = cv2.resize(frame, (640, 360))

        # Extract tracking info
        frame, track_ids = extract_track_info(frame, frame_counter)

        # Update track history
        for track_id in track_ids:
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(frame_counter)

        # Display frame with tracking info
        cv2_imshow(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

    # Post-processing and statistics
    total_tracks = len(track_history)
    track_lifespans = [
        max(frames) - min(frames) + 1 for frames in track_history.values()
    ]

    print(f"Total tracks: {total_tracks}")
    if total_tracks > 0:
        print(
            f"Average track lifespan: {sum(track_lifespans) / len(track_lifespans):.2f} frames"
        )
        print(f"Maximum track lifespan: {max(track_lifespans)} frames")
    else:
        print("No tracks detected.")

    # Plot metrics
    plot_metrics(track_history)

if __name__ == "__main__":
    main()
