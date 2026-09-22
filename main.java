# Step 1: Install required libraries
!pip install ultralytics opencv-python-headless matplotlib

import cv2
import urllib.request
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Step 2: Download a sample crowd image
img_url = "https://images.pexels.com/photos/1097456/pexels-photo-1097456.jpeg"

# Add User-Agent header to avoid 403 Forbidden error
req = urllib.request.Request(
    img_url,
    data=None,
    headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    }
)

with urllib.request.urlopen(req) as response, \
        open("crowd.jpg", 'wb') as out_file:
    out_file.write(response.read())

# Step 3: Load pre-trained YOLOv8 model and run detection on people (class 0)
model = YOLO('yolov8n.pt')
results = model("crowd.jpg", classes=[0])

# Step 4: Render bounding boxes on the image
annotated_frame = results[0].plot()

# Step 5: Save and display the result
output_path = "crowd_detection_result.jpg"
cv2.imwrite(output_path, annotated_frame)

# Display in Colab
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f"Crowd Density Detection: {len(results[0].boxes)} People Detected")
plt.show()
