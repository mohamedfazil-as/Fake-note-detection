import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/fake_note_detector.h5'

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("model", exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found at: {os.path.abspath(MODEL_PATH)}")


# Function to preprocess images
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("❌ Error: Image not loaded correctly.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (128, 128))  # Match training size
        img = img.astype(np.float32) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        print(f"✅ Image processed successfully: {image_path}")
        return img
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None


@app.route("/", methods=["GET"])
def home():
    return render_template("index1.html")


@app.route("/upload_predict", methods=["POST"])
def upload_predict():
    if model is None:
        return "❌ Model not loaded. Please check logs.", 500

    if "file" not in request.files:
        return "❌ No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "❌ No selected file", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess and make a prediction
    image = preprocess_image(file_path)
    if image is None:
        return "❌ Error processing image", 500

    prediction = model.predict(image)
    print(f"🔍 Model Prediction Output: {prediction}")  # Debugging log

    # Ensure output is scalar
    if isinstance(prediction, (list, np.ndarray)):
        prediction = prediction[0][0]

    # Adjusted threshold
    THRESHOLD = 1  # Adjust based on model performance
    result = "✅ Real Note" if prediction > THRESHOLD else "❌ Fake Note"
    confidence = round(float(prediction) * 100, 2)

    print(f"🔍 Final Classification: {result} (Confidence: {confidence}%)")

    return render_template("result1.html", result=result, accuracy=confidence)


if __name__ == "__main__":
    app.run(debug=True)
