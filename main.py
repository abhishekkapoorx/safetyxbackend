import pandas as pd
import cv2
from ultralytics import YOLO
from flask import Flask, jsonify, request
import os

# Create a Flask application
app = Flask(__name__)

# Load your trained YOLO model
model = YOLO("best.pt")  # Replace with your trained model path

# Print model class names to debug
print("Model class names:", model.names)

app.config['UPLOAD_FOLDER'] = "uploads/"

# Define a function to process video and apply detection boxes
def process_video(input_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize list to store outsider detection timestamps
    outsider_timestamps = []
    
    # Initialize counters for each class
    class_counts = {class_name: 0 for class_name in model.names.values()}  # Create a dictionary for class counts

    # Initialize frame counter
    frame_counter = 0
    
    # Create a list to store counts for each second
    detections_per_second = {}

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop when the video ends

        frame_counter += 1  # Increment frame counter

        # Get the current timestamp in seconds
        current_second = int(frame_counter / fps)

        # Process every 60th frame
        if frame_counter % 60 == 0:
            # Run inference on the current frame
            results = model([frame])  # Inference on the single frame

            # Initialize a dictionary to store counts for the current second
            detected_counts = {class_name: 0 for class_name in model.names.values()}

            # Process each result in the batch
            for result in results:
                # Loop through each detected box
                for box in result.boxes:
                    class_idx = int(box.cls)  # Get the class index from the detected box
                    class_name = model.names[class_idx]  # Convert the class index to class name (string)

                    # Check if the detection is above a confidence threshold (e.g., 0.5)
                    if box.conf > 0.5:
                        # Increment count for the detected class
                        class_counts[class_name] += 1
                        detected_counts[class_name] += 1  # Increment for the current second
                        
                        # Check if the class is 'outsider' and save timestamp
                        if class_name == 'outsider':
                            outsider_timestamps.append(current_second)

            # Store detected counts for the current second in the dictionary
            detections_per_second[current_second] = detected_counts

    # Release everything when done
    cap.release()

    # Remove duplicates and round timestamps to two decimal places for 'outsider' detections
    outsider_timestamps = sorted(set(round(ts, 2) for ts in outsider_timestamps))

    # Create a JSON-friendly output
    output = {
        "timestamps": outsider_timestamps,
        "total_counts": class_counts,
        "detections_per_second": detections_per_second
    }
    
    return output

# Define an API route to process video
@app.route('/process_video', methods=['POST'])
def video_processing_api():
    # Get the video path from the request
    video_path = request.files.get('video_path')  # Expecting JSON with {"video_path": "path/to/video.mp4"}

    if video_path:
        # Save the file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], video_path.filename)
        video_path.save(filepath)

    if not video_path:
        return jsonify({"error": "No video path provided"}), 400

    try:
        # Process the video and get results
        results = process_video(filepath)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/')
def home():
    return "Welcome to the YOLO API!"

if __name__ == '__main__':
    app.run(debug=True)