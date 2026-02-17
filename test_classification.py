import requests
import os
import json

# Find a test image
test_image_path = None
for root, dirs, files in os.walk("dataset"):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_image_path = os.path.join(root, file)
            break
    if test_image_path:
        break

if test_image_path:
    print(f"Found test image: {test_image_path}")
    with open(test_image_path, 'rb') as f:
        # Send as multipart form data (application/x-www-form-urlencoded)
        files = {'file': (os.path.basename(test_image_path), f, 'image/jpeg')}
        try:
            response = requests.post("http://localhost:5000/api/predict", files=files, timeout=30)
            print(f"Response status: {response.status_code}")
            result = response.json()
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
else:
    print("No test image found")
