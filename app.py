# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image
import cv2

app = Flask(__name__)

# Load the pre-trained model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image_data']
        
        # Remove the data:image/png;base64, prefix
        image_data = image_data.split(',')[1]
        
        # Decode the base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 first for better processing
        image = image.resize((28, 28))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Invert the image (black background, white digit)
        img_array = 255 - img_array
        
        # Threshold to remove noise and background
        _, img_array = cv2.threshold(img_array, 50, 255, cv2.THRESH_BINARY)
        
        # Find the center of mass
        moments = cv2.moments(img_array)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
        else:
            cX, cY = 14, 14  # Default center
        
        # Create a new square image with padding
        size = 28
        new_img = np.zeros((size, size), dtype=np.uint8)
        
        # Calculate the bounding box
        coords = cv2.findNonZero(img_array)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Center the digit
            offsetX = int((size - w) / 2)
            offsetY = int((size - h) / 2)
            
            # Create a region of interest
            for i in range(h):
                for j in range(w):
                    if y+i < img_array.shape[0] and x+j < img_array.shape[1]:
                        if img_array[y+i, x+j] > 0:
                            newY = offsetY + i
                            newX = offsetX + j
                            if 0 <= newY < size and 0 <= newX < size:
                                new_img[newY, newX] = img_array[y+i, x+j]
        
        # Resize to 8x8 for the model
        img_8x8 = cv2.resize(new_img, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Normalize the pixel values to match the example data range (0-16)
        img_normalized = img_8x8.astype(float) / 255.0 * 16.0
        
        # Convert to the feature array format
        features = []
        for i in range(8):
            for j in range(8):
                features.append(float(img_normalized[i, j]))
        
        # Print for debugging
        print("Processed features:", features)
        
        # Make prediction
        digit_prediction = model.predict([features])[0]
        
        return jsonify({'prediction': int(digit_prediction), 'success': True})
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e), 'success': False})

if __name__ == '__main__':
    app.run(debug=True)