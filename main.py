from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image 

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model_path = './leaf_classfi.h5'
model = load_model(model_path)

# Define the class labels (make sure these match your model's output classes)
class_labels = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'ashoka', 'Astma_weed', 
                'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 
                'Bringaraja', 'camphor', 'Caricature', 'Castor', 'Catharanthus', 
                'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 
                'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddapthre', 
                'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 
                'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 
                'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 
                'kamakasturi', 'Kambajala', 'Kasambruga', 'kepala', 'Kohlrabi', 
                'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 
                'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 
                'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 
                'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 
                'Pomegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 
                'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 
                'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric']


@app.get("/") 
def home():
    return jsonify({"message":"Hello"} ) 

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is part of the POST request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Load and preprocess the image
    img = Image.open(file.stream) 
    img = img.resize((224, 224)) 
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]

    # Return the prediction as a JSON response
    return jsonify({"class": predicted_class, "confidence": float(np.max(prediction))})

# Run the app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        # logging.error("An error occurred while running the Flask app", exc_info=True)
        print(e) 