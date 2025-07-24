import torch
import torch.nn as nn
import numpy as np
import json
import os

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

model = None

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "mnist_mlp_model.pth")
    checkpoint = torch.load(model_path, map_location="cpu")

    model = MLP(
        input_size=checkpoint['model_architecture']['input_size'],
        hidden_sizes=checkpoint['model_architecture']['hidden_sizes'],
        num_classes=checkpoint['model_architecture']['num_classes'],
        dropout_rate=checkpoint['model_architecture']['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

def run(raw_data):
    try:
        data = json.loads(raw_data)
        image = np.array(data["image"], dtype=np.float32)
        image = (image - 0.1307) / 0.3081
        tensor = torch.FloatTensor(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1).numpy()[0]
            predicted = int(np.argmax(probs))

        return {
            "predicted_digit": predicted,
            "confidence": round(probs[predicted] * 100, 2),
            "probabilities": [round(p * 100, 2) for p in probs]
        }
    except Exception as e:
        return {"error": str(e)}
