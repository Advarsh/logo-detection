from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the saved model
model = load_model('model.h5')

# Define the class names
class_names = ['adidas', 'apple', 'bmw', 'citroen', 'cocacola', 'dHL', 'fedex', 'ferrari', 'ford', 'google', 'heineken', 'hp', 'mcdonalds', 'mercedes', 'nike', 'pepsi', 'porsche', 'puma', 'red_bull', 'shell', 'singapore_airlines', 'starbucks', 'ups', 'visa', 'volkswagen', 'yahoo', 'ysl']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the HTML form
    file = request.files['image']
    
    # Preprocess the image
    img = Image.open(file.stream)
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make a prediction using the loaded model
    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    class_name = class_names[class_idx]
    score = prediction[class_idx]
    
    # Render the result HTML template with the predicted class name and score
    return render_template('result.html', class_name=class_name, score=score)

if __name__ == '__main__':
    app.run(debug=True)