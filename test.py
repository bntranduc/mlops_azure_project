import requests

with open("/home/bao/ESGI/4IABD/S2/mlops_azure_project/images.png", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)

print(response.json())
