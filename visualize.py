import requests
import cv2
import numpy as np
from PIL import Image
import io

# 1. Send image to your running app.py
url = "http://127.0.0.1:8000/predict"
img_path = "BCCD_Dataset-master/BCCD_Dataset-master/BCCD/JPEGImages/BloodImage_00003.jpg"

with open(img_path, "rb") as f:
    response = requests.post(url, files={"file": f})
    data = response.json()

# 2. Load the image with OpenCV for drawing
img = cv2.imread(img_path)
label_map = {1: "WBC", 2: "RBC", 3: "Platelets"}

# 3. Draw the boxes returned by the API
for box, label, score in zip(data['boxes'], data['labels'], data['scores']):
    xmin, ymin, xmax, ymax = map(int, box)
    
    # Draw Rectangle
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    # Add Label Text
    text = f"{label_map.get(label, 'Unknown')}: {score:.2f}"
    cv2.putText(img, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 4. Show the result
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()