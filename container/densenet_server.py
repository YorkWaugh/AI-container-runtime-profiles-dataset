# densenet_server.py
import torch
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms
import io
import time

model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights='IMAGENET1K_V1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() 
print(f"Model loaded on {device}")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = Flask(__name__)

@app.route("/infer", methods=['POST'])
def infer():
    data = request.get_json()
    
    input_tensor = torch.tensor(data['tensor']).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
    latency = time.perf_counter() - t0
    
    return jsonify({
        "status": "success",
        "latency": latency
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8006, debug=False)