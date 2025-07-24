from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Tuple
import numpy as np
from PIL import Image
import io
import os
import requests

app = FastAPI(title="MNIST Digit Recognition API via Azure")

# === Configuration Azure ML ===
SCORING_URI = os.getenv("AZURE_ML_URI")  # Ex: "https://xxx.inference.ml.azure.com/score"
API_KEY = os.getenv("AZURE_ML_KEY")      # Clé primaire

# === Preprocessing ===
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = Image.eval(img, lambda x: 255 - x)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array

# === Azure ML Call ===
def predict_via_azure(img_array: np.ndarray):
    payload = {
        "image": img_array.tolist()
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(SCORING_URI, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur Azure ML : {response.status_code} - {response.text}")

# === FastAPI Endpoint ===
@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    if not file.filename.endswith(("png", "jpg", "jpeg")):
        return JSONResponse(status_code=400, content={"error": "Fichier image requis"})

    image_bytes = await file.read()
    try:
        img_array = preprocess_image(image_bytes)
        result = predict_via_azure(img_array)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
