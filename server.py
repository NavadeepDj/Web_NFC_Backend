from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your YOLOv8 model
model = YOLO('best (15).pt')  # Replace with your model path

@app.route('/detect_person', methods=['POST'])
def detect_person():
    try:
        # Get the image data from the request
        data = request.json
        image_data = data['image']
        
        # Convert base64 image to PIL Image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run inference on the image
        results = model(image)  # Run the YOLO model
        
        # Get the first result (assuming single image input)
        result = results[0]
        
        # Convert the result to a numpy array for drawing boxes
        img = np.array(image)
        
        # List to store detected persons
        detected_persons = []
        
        # Loop through detected boxes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
            conf = float(box.conf[0])  # Get confidence score
            cls = int(box.cls[0])  # Get class id
            
            # Check if the class id corresponds to 'person' (usually 0 in COCO dataset)
            if cls == 0:  # 'person' class in COCO
                detected_persons.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
                
                # Prepare label for the bounding box
                label = f"Person {conf:.2f}"
                
                # Draw bounding box and label on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert image back to base64 for response
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Return response with detected persons and the modified image
        return jsonify({
            'success': True,
            'detected_persons': detected_persons,
            'image': img_base64  # Return the image with bounding boxes as base64
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0')
