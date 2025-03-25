from django.shortcuts import render

# Create your views here.
import onnxruntime as rt
import numpy as np
import joblib
import pandas as pd
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
GOOGLE_GEMINI_API_KEY = "AIzaSyDAQ1HL4v7XEWRzkzMpxFm_1WsaYyN7sfE"

import gdown
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model1.onnx")
CSV_PATH = os.path.join(BASE_DIR, "models", "Final_Augmented_dataset_Diseases_and_Symptoms.csv")

# Google Drive File IDs (Replace with your actual file IDs)
MODEL_FILE_ID = "1Ohx7LM1c3avKXONJIiVKFTMtQisiqqLi"
CSV_FILE_ID = "1prdAX9PU_J6ml-vjFJSuPKHXZcTUR_8u"

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive if it doesn't exist."""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded: {output_path}")


# Ensure models directory exists
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Download files
download_from_gdrive(MODEL_FILE_ID, MODEL_PATH)
download_from_gdrive(CSV_FILE_ID, CSV_PATH)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model1.onnx")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder1.pkl")
# CSV_PATH = os.path.join(BASE_DIR, "models", "Final_Augmented_dataset_Diseases_and_Symptoms.csv")

sess = rt.InferenceSession(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
data = pd.read_csv(CSV_PATH)
all_symptoms = list(data.columns.drop("diseases"))  # List of symptoms used in training

# Configure Gemini AI
genai.configure(api_key=settings.GOOGLE_GEMINI_API_KEY)
from django.views.decorators.csrf import csrf_exempt

def predict_disease(user_symptoms):
    """Predict disease from user symptoms using ONNX model."""
    sample_input = np.zeros((1, len(all_symptoms)), dtype=np.float32)

    print("User Symptoms:", user_symptoms)  # Debugging

    for symptom in user_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            sample_input[0, index] = 1
            print(f"Setting {symptom} (index {index}) to 1")  # Debugging

    print("Input to Model:", sample_input)  # Debugging

    # Get model input name
    input_name = sess.get_inputs()[0].name

    # Run prediction
    prediction = sess.run(None, {input_name: sample_input})[0]

    print("Model Output:", prediction)  # Debugging

    # Decode the predicted disease
    # Ensure prediction is an array of probabilities
    if prediction.ndim > 1:
        predicted_index = np.argmax(prediction[:5])  # Extract first row if 2D
    else:
        predicted_index = int(prediction[0])  # Access the first element explicitly  # If already a class index, use directly
    predicted_disease = encoder.inverse_transform([predicted_index])[0]

    print("Predicted Disease:", predicted_disease)  # Debugging
    print("Prediction Probabilities:", prediction[0])  # ✅ Debug model output

    return predicted_disease


def generate_advice(disease):
    """Generate medical advice using Gemini AI."""
    prompt = f"What are the symptoms and possible treatments for {disease}?"
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

import re
@csrf_exempt
def disease_prediction(request):
    if request.method == "POST":
        user_input = request.POST.get("symptoms", "")  # Get symptoms from user text input
        
        if not user_input:
            return JsonResponse({"error": "No symptoms provided."})

        # Extract symptoms from the input text
        user_symptoms = extract_symptoms(user_input)

        if not user_symptoms:
            return JsonResponse({"error": "No known symptoms detected. Try again with different wording."})

        # Predict disease
        predicted_disease = predict_disease(user_symptoms)

        # Get AI-generated advice
        medical_advice = generate_advice(predicted_disease)

        return JsonResponse({"disease": predicted_disease, "advice": medical_advice})

    return render(request, "disease_form.html")  # A template for the form

def extract_symptoms(user_input):
    """Extract symptoms from user input by matching with predefined symptom list."""
    user_input = user_input.lower()  # Convert input to lowercase
    symptoms_found = []

    for symptom in all_symptoms:
        if re.search(rf"\b{re.escape(symptom)}\b", user_input):
            symptoms_found.append(symptom)

    return symptoms_found







from .models import UploadedImage
import pytesseract
from PIL import Image
from django.core.files.storage import FileSystemStorage

# Set Tesseract-OCR path (Only required for Windows)
pytesseract.pytesseract.tesseract_cmd = r"D:\Hp\Program Files\Tesseract-OCR\tesseract.exe"
# Verify it's accessible
print("Tesseract Path:", pytesseract.pytesseract.tesseract_cmd)
def home(request):
    return render(request, 'home.html')

import cv2
import numpy as np

def preprocess_image(image_path):
    """Convert image to grayscale and apply thresholding for better OCR."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, processed_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    return processed_image

def extract_text(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        
        # Save file
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        # Process image with OpenCV
        processed_image = preprocess_image(fs.path(file_path))

        # Extract text
        extracted_text = pytesseract.image_to_string(processed_image)

        # Save result in database
        uploaded_image = UploadedImage.objects.create(image=uploaded_file, extracted_text=extracted_text)

        return render(request, 'home.html', {'image_url': file_url, 'extracted_text': extracted_text})

    return render(request, 'home.html')
