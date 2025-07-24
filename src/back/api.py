from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io

app = FastAPI(title="MNIST Digit Recognition API")

# === MLP Definition ===
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

# === Load Model ===
def load_model(model_path="mnist_mlp_model.pth"):
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device)

    model = MLP(
        input_size=checkpoint['model_architecture']['input_size'],
        hidden_sizes=checkpoint['model_architecture']['hidden_sizes'],
        num_classes=checkpoint['model_architecture']['num_classes'],
        dropout_rate=checkpoint['model_architecture']['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()

# === Preprocessing ===
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = Image.eval(img, lambda x: 255 - x)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.1307) / 0.3081
    return img_array

# === Prediction ===
def predict(img_array: np.ndarray) -> Tuple[int, np.ndarray]:
    tensor = torch.FloatTensor(img_array).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        predicted = int(np.argmax(probs))
    return predicted, probs

# === FastAPI Route ===
@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    if not file.filename.endswith(("png", "jpg", "jpeg")):
        return JSONResponse(status_code=400, content={"error": "Fichier image requis"})
    
    image_bytes = await file.read()
    try:
        img_array = preprocess_image(image_bytes)
        predicted_class, probabilities = predict(img_array)
        return {
            "predicted_digit": predicted_class,
            "confidence": round(probabilities[predicted_class] * 100, 2),
            "probabilities": [round(p * 100, 2) for p in probabilities]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
