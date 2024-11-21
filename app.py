from flask import Flask, render_template, request
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import os

app = Flask(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Use GPU for faster performance

# Load your trained YOLO models for sticker and serial number detection
sticker_model = YOLO('./models/sticker_detection_model.pt')  # Path to your YOLO sticker detection model
serial_number_model = YOLO('./models/yolosno.pt')  # Path to your YOLO serial number detection model

# Function to get the rotation angle using Hough Line Transform
def get_rotation_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # Detect edges using Canny edge detection

    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # Detect lines in the image

    # If lines are detected, calculate the rotation angle
    if lines is not None:
        rho, theta = lines[0][0]
        angle = np.degrees(theta) - 90  # Convert angle to degrees
        return angle
    return 0  # Return 0 if no lines are detected (no rotation)

# Function to rotate the image based on the detected angle
def rotate_image(image, angle):
    if angle == 0:
        return image  # No rotation needed

    (h, w) = image.shape[:2]  # Get the height and width of the image
    center = (w // 2, h // 2)  # Find the center of the image
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # Get the rotation matrix
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # Apply rotation with black border
    return rotated_image

# Function to detect serial number region
def detect_serial_number(cropped_rotated_image):
    # Predict serial number region using the YOLO model
    results = serial_number_model.predict(cropped_rotated_image)

    # Get bounding boxes, classes, and confidence scores
    boxes = results[0].boxes
    if len(boxes) > 0:
        box = boxes.xyxy[0]  # Get the first bounding box for serial number
        x1, y1, x2, y2 = box  # Coordinates of the detected serial number region
        serial_number_crop = cropped_rotated_image[int(y1):int(y2), int(x1):int(x2)]  # Crop the serial number region
        return serial_number_crop
    return None  # Return None if no serial number is detected

# Function to extract serial number using EasyOCR
def extract_serial_number(image):
    # Use EasyOCR to extract text
    results = reader.readtext(image, detail=0)  # Get only text without bounding boxes
    for text in results:
        # Filter for text containing digits (4-10 length, assumes it's the serial number)
        if text.isdigit() and 4 <= len(text) <= 10:
            return text  # Return the detected serial number
    return "not detected"

# Function to process the image using YOLO and then apply rotation to the cropped regions
def process_image(image):
    # Make predictions with YOLOv8 model (sticker detection)
    results = sticker_model.predict(image)

    # Get bounding boxes, classes, and confidence scores
    boxes = results[0].boxes
    scores = boxes.conf
    classes = boxes.cls

    cropped_serial_images = []  # List to store cropped serial number images
    detected_serial_number = "not detected"

    # For each detected object, crop the image and rotate
    for box, score, cls in zip(boxes.xyxy, scores, classes):
        x1, y1, x2, y2 = box
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]  # Crop the image
        angle = get_rotation_angle(cropped_image)  # Get the rotation angle for the cropped image
        rotated_cropped_image = rotate_image(cropped_image, angle)  # Rotate the cropped image

        # Detect serial number from the rotated cropped image
        serial_number_image = detect_serial_number(rotated_cropped_image)
        if serial_number_image is not None:
            cropped_serial_images.append(serial_number_image)  # Add detected serial number image to list
            # Extract serial number text using EasyOCR
            detected_serial_number = extract_serial_number(serial_number_image)
            break  # Stop after the first valid detection

    return detected_serial_number, cropped_serial_images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # Save the uploaded image to a temporary location
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Load the image
    image = cv2.imread(file_path)

    # Process the image
    detected_serial_number, cropped_serial_images = process_image(image)

    # Save the original and cropped serial images as static files to display in the webpage
    original_image_path = os.path.join('static', 'original_image.png')
    cv2.imwrite(original_image_path, image)

    cropped_image_paths = []
    for idx, serial_image in enumerate(cropped_serial_images):
        cropped_image_path = os.path.join('static', f'cropped_serial_{idx}.png')
        cv2.imwrite(cropped_image_path, serial_image)
        cropped_image_paths.append(cropped_image_path)

    return render_template('index.html', 
                           serial_number=detected_serial_number, 
                           original_image_path=original_image_path, 
                           cropped_images=cropped_image_paths)

if __name__ == '__main__':
    app.run(debug=True)
