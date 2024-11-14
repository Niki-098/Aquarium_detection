import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model = YOLO('best.pt')  

# Streamlit app configuration
st.title("YOLOv8 Object Detection")
st.write("Upload an image to perform object detection.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Converting the uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Performing detection
    results = model.predict(image_np)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0]  # Get confidence score
            class_id = int(box.cls[0])  # Get class ID
            class_name = model.names[class_id]  # Get class name

            # Draw the bounding box and label on the image
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Converting the image back to PIL format for display
    result_image = Image.fromarray(image_np)
    st.image(result_image, caption="Detected Image", use_column_width=True)
