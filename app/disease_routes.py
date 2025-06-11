from flask import Blueprint, render_template, request
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# Initialize the blueprint for the disease detection module
disease_bp = Blueprint('disease', __name__)

# Load the trained model (ensure the model file path is correct)
MODEL_PATH = r"Best_Plant.h5"  # Path to the model file
model = load_model(MODEL_PATH)

IMG_SIZE = 128  # Resize image to this size before prediction

# Solution dictionary mapping diseases to recommended solutions
disease_solutions = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant hybrids, crop rotation, and fungicide sprays.",
    "Corn_(maize)___Common_rust_": "Apply fungicides if severe; use resistant varieties.",
    "Corn_(maize)___healthy": "No action needed. Keep monitoring the crop.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids and apply fungicides early.",
    "Potato___Early_blight": "Remove infected leaves, rotate crops, and apply appropriate fungicides.",
    "Potato___healthy": "No disease detected. Maintain proper care.",
    "Potato___Late_blight": "Remove infected plants and spray systemic fungicides.",
    "Tomato___Bacterial_spot": "Use copper-based sprays and resistant varieties.",
    "Tomato___Early_blight": "Use mulch, remove infected leaves, and apply fungicides.",
    "Tomato___healthy": "No disease detected. Continue regular care.",
    "Tomato___Late_blight": "Apply fungicides and remove infected leaves immediately.",
    "Tomato___Leaf_Mold": "Improve air circulation, avoid overhead watering, and use fungicides.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply chlorothalonil or copper fungicide.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray with water or insecticidal soap. Introduce natural predators.",
    "Tomato___Target_Spot": "Use crop rotation, remove debris, and apply fungicides.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect tools. Avoid handling when wet.",
    "Tomato___Yellow_Leaf_Curl_Virus": "Control whiteflies and use resistant plant varieties."
}

# Route for disease detection
@disease_bp.route('/', methods=['GET', 'POST'])
def disease_detection():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('disease_interface.html', error="No file part")

        file = request.files['image']
        if file.filename == '':
            return render_template('disease_interface.html', error="No selected file")

        try:
            # Read and process the uploaded image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to fit model input
            img_array = img_to_array(img_resized) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])  # Get the index of the highest probability
            class_labels = list(disease_solutions.keys())  # Disease classes
            result = class_labels[predicted_class]  # Get the corresponding class label
            solution = disease_solutions.get(result, "No solution available.")  # Get the corresponding solution

            # Return result and solution to the template
            return render_template('disease_interface.html', result=result, solution=solution)

        except Exception as e:
            # Log the error and display a user-friendly message
            print(f"Error during prediction: {e}")
            return render_template('disease_interface.html', error="Error: Something went wrong with the prediction.")

    # If GET request, just render the empty page
    return render_template('disease_interface.html')
