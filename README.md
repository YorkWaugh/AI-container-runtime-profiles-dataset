# AI Model Container Runtime Profiling (AC-Prof) Dataset
>
> Reproducible measurements of the invocation latency of AI Services Docker Containers, including cold starts and runtime behavior, under various resource specifications and input scales.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#requirements) [![Issues](https://img.shields.io/github/issues/wingter562/AI-container-runtime-profiles-dataset.svg)](https://github.com/wingter562/AI-container-runtime-profiles-dataset/issues)

## Overview

This repository provides

1. A dataset of latency measurements for popular AI service containers (with deep models at the core)
2. Scripts for systematically profiling containerized ML workloads.
We focus on two critical aspects for scheduling and resource management: container cold start and nonlinear runtime behavior under varid CPU, GPU, and memory specs as well as input sizes. The dataset is designed to support reproducible performance modeling and quantitative evaluation of scheduling and resource allocation strategies.

## What’s Included

- **Static metrics**: container image and model weights size (download volume).
- **Dynamic metrics**: end-to-end runtime measured under a specified matrix of CPU/GPU/memory limits and input scales.
- **Optional metrics**: peak CPU usage, peak GPU utilization or VRAM usage.
- **Scripts**: Docker build/run recipes with resource caps, client scripts for warm-up and request timing, CSV logging, and plotting utilities.

## Measurement Environment

- OS: Ubuntu 24.04.6 LTS  
- Container runtime: Docker 27.5.1  
- Drivers/Libraries: CUDA 12.1, cuDNN 9.1  
- Language/Framework: Python 3.12, PyTorch 2.5.1+cu121

## Data Sources

This dataset is collected with reference to the **APIBench** dataset methodology. External model APIs are sourced from three popular ML model repositories:

- **TorchHub**: <https://pytorch.org/hub/>
- **TensorFlow Hub**: <https://www.tensorflow.org/hub>
- **HuggingFace Models**: <https://huggingface.co/models>

## Resource and Input Matrix

- CPU cores: `{1, 2, 4, 8}`  
- Memory limits: `{2 GB, 4 GB, 8 GB, 16 GB}`  
- GPU count: `{0, 1}`  
- Input scaling: task-specific multi-level inputs (e.g., image resolution or text length grid)

## Data Example

![runtime profile example](docs/container_runtime_example.png)

## Guideline for Customized Profiling

### Requirements

- Python 3.10 or newer
- Git and a recent C or C++ toolchain if native dependencies are required
- Optional CUDA 12.x for GPU features

### What to Record

- Startup metrics: image download size and time, container initialization time,.
- Runtime metrics: per-request end-to-end latency under each CPU/MEM/GPU/input configuration.
- Optional peaks: peak CPU usage, peak GPU utilization/VRAM.

### Quick Start Tutorial: profiling FCN-50

#### 1) Model Download (download_model.py)

```python
import os
import torch
import torchvision
import shutil

# Set local storage directory (ensure models are downloaded here)
TORCH_HOME = "data/torch_hub"
os.makedirs(TORCH_HOME, exist_ok=True)  # Ensure the directory exists

# Only effective within the current process, does not affect the global `TORCH_HOME`
torch.hub.set_dir(TORCH_HOME)

# Model information
MODEL_REPO = "pytorch/vision"
MODEL_NAME = "fcn_resnet50"
PRETRAINED = True

print(f"Pre-downloading model: {MODEL_NAME}, repository: {MODEL_REPO}, storage path: {TORCH_HOME}")
model = torch.hub.load(repo_or_dir=MODEL_REPO, model=MODEL_NAME, pretrained=PRETRAINED)
print(f"Model pre-download complete, storage path: {TORCH_HOME}")
```

#### 2) Write inference service code（server.py）

```python
import torch
import torchvision
from flask import Flask, request, jsonify
from codecarbon import EmissionsTracker
import os
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

app = Flask(__name__)

# ---------------------------
# 1. Model Loading
# ---------------------------
# Set TORCH_HOME environment variable to tell torchvision where to find pre-downloaded weights.
# This must be set before loading the model.
os.environ['TORCH_HOME'] = '/opt/torch_hub'

print("Loading model from installed torchvision...")

# Load model using the official library.
# weights='DEFAULT' loads the latest pretrained weights (fcn_resnet50_coco-*.pth).
# For older PyTorch versions, use pretrained=True.
try:
    # New torchvision syntax (0.13+)
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
except ImportError:
    # Legacy compatibility
    print("Using legacy loading method...")
    model = fcn_resnet50(pretrained=True)

model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Model loaded on {device}")

# ---------------------------
# 2. Initialize CodeCarbon Tracker
# ---------------------------
tracker = EmissionsTracker(
    output_dir="/opt/logs", 
    output_file="emissions.csv", 
    on_csv_write="append",
    measure_power_secs=0.1,
    save_to_file=True,
    log_level="error"
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if "tensor" not in data:
            return jsonify({"error": "No tensor provided"}), 400
            
        input_tensor = torch.tensor(data['tensor'])
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # ---------------------------
        # 3. Start Energy Tracking
        # ---------------------------
        tracker.start()
        
        with torch.no_grad():
            output = model(input_tensor)
            
        # ---------------------------
        # 4. Stop Tracking
        # ---------------------------
        tracker.stop()
        
        # Calculate energy consumption in Joules
        emissions_data = tracker.final_emissions_data
        kwh_to_joules = 3.6e6
        
        cpu_energy = emissions_data.cpu_energy * kwh_to_joules
        gpu_energy = emissions_data.gpu_energy * kwh_to_joules
        ram_energy = emissions_data.ram_energy * kwh_to_joules
        total_energy = emissions_data.energy_consumed * kwh_to_joules

        # Convert output to list
        if isinstance(output, dict):
            res_numpy = output['out'].cpu().numpy().tolist()
        else:
            res_numpy = output.cpu().numpy().tolist()

        return jsonify({
            "status": "success",
            "segmentation": res_numpy, 
            "energy_metrics": {
                "cpu_joules": cpu_energy,
                "gpu_joules": gpu_energy,
                "ram_joules": ram_energy,
                "total_joules": total_energy
            }
        })

    except Exception as e:
        if tracker._active:
            tracker.stop()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8006)
```

#### 3) Compose the Dockerfile

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install Python dependencies
# Note: torch and torchvision are already included in the base image, so they are not listed here.
RUN pip install --default-timeout=600 \
    pytorchvideo \
    flask opencv-python-headless \
    codecarbon \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV TORCH_HOME=/opt/torch_hub

WORKDIR /opt

# Copy scripts and pre-downloaded model data
COPY download_model.py server.py /opt/
COPY data/torch_hub/ /opt/torch_hub/

# Create directory for logs
RUN mkdir -p /opt/logs

EXPOSE 8006

CMD ["python", "server.py"]
```

#### 4) Deploy containers with specified resource budgets

```bash
sudo docker run --gpus all \
  -d -p 9001:8006 \
  --cpus="8.0" --memory="16g" \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  --name fcn_resnet50_gpu \
  torchhub_fcn_resnet50_server

sudo docker run \
  -d -p 9002:8006 \
  --cpus="6.0" --memory="12g" \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  --name fcn50_cpu6 \
  torchhub_fcn_resnet50_server

sudo docker run \
  -d -p 9003:8006 \
  --cpus="4.0" --memory="8g" \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  --name fcn50_cpu4 \
  torchhub_fcn_resnet50_server

sudo docker run \
  -d -p 9004:8006 \
  --cpus="3.0" --memory="6g" \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  --name fcn50_cpu3 \
  torchhub_fcn_resnet50_server

sudo docker run \
  -d -p 9005:8006 \
  --cpus="2.0" --memory="4g" \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  --name fcn50_cpu2 \
  torchhub_fcn_resnet50_server

sudo docker run \
  -d -p 9006:8006 \
  --cpus="1.0" --memory="2g" \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  --name fcn50_cpu1 \
  torchhub_fcn_resnet50_server
```

#### 5）Metrics collection code (setting different input images can collect runtime metrics given different input sizes)

```python
import requests
import numpy as np
import cv2
import time
import json
import csv
from PIL import Image
from torchvision import transforms

def load_local_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image, please check the path: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def test_inference(api_url, img, orig_size):
    img_pil = Image.fromarray(img)
    input_tensor = preprocess(img_pil)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    payload = {
        "tensor": input_tensor.cpu().numpy().tolist(),
        "orig_size": orig_size
    }
    
    start = time.perf_counter()
    try:
        response = requests.post(api_url, json=payload, timeout=60)
        elapsed = time.perf_counter() - start
        return response, elapsed
    except Exception as e:
        print(f"Request Error: {e}")
        return None, time.perf_counter() - start

def main():
    # TODO: Please modify this to your local image path
    image_path = "test.jpg" 
    try:
        original_img = load_local_image(image_path)
    except ValueError as e:
        print(e)
        return

    orig_size = [original_img.shape[0], original_img.shape[1]]
    # Ensure port matches the docker run mapping (e.g., 9001 -> 8006)
    api_url = "http://localhost:9001/predict" 
    
    # ---------------------------
    # Warm-up
    # ---------------------------
    print("Warming up...")
    resp, warmup_time = test_inference(api_url, original_img, orig_size)
    if resp and resp.status_code == 200:
        print(f"Warm-up call took: {warmup_time:.4f} seconds")
    else:
        print("Warm-up failed.")
    
    scales = np.linspace(0.1, 2.0, 20)
    results = []
    
    print("\nStarting energy and latency profiling...")
    for s in scales:
        new_width = int(original_img.shape[1] * s)
        new_height = int(original_img.shape[0] * s)
        
        # Avoid zero dimensions
        if new_width < 1 or new_height < 1: continue

        img_scaled = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        response, elapsed = test_inference(api_url, img_scaled, orig_size)
        
        cpu_j = 0.0
        gpu_j = 0.0
        
        if response and response.status_code == 200:
            data = response.json()
            # Get energy metrics from server response
            energy_metrics = data.get("energy_metrics", {})
            cpu_j = energy_metrics.get("cpu_joules", 0.0)
            gpu_j = energy_metrics.get("gpu_joules", 0.0)
            
            print(f"Scale {s:.2f} | Time: {elapsed:.4f}s | CPU: {cpu_j:.4f}J | GPU: {gpu_j:.4f}J")
        else:
            print(f"Scale {s:.2f} | Failed Status: {response.status_code if response else 'Error'}")

        results.append({
            "scale": s,
            "width": new_width,
            "height": new_height,
            "time": elapsed,
            "cpu_joules": cpu_j,
            "gpu_joules": gpu_j
        })
    
    # Save results including Energy metrics
    with open("inference_profile.csv", "w", newline="") as csvfile:
        fieldnames = ["scale", "width", "height", "time", "cpu_joules", "gpu_joules"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({
                "scale": f"{res['scale']:.4f}",
                "width": res["width"],
                "height": res["height"],
                "time": f"{res['time']:.4f}",
                "cpu_joules": f"{res['cpu_joules']:.6f}",
                "gpu_joules": f"{res['gpu_joules']:.6f}"
            })
    print("\nResults saved to inference_profile.csv")

if __name__ == '__main__':
    main()
```

### Modeling Guidance

- After collecting measurements for a container, fit a simple parametric or piecewise model (e.g., least squares) for latency as a function of resources and input size, and report goodness of fit and residuals. Keep train/test splits separate for each container–task pair.

## Contribution

New contributors are welcome. Please open an issue to discuss your idea before submitting a pull request. Follow the code style and ensure tests pass. See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` if present.

## License

This project is released under the Apache-2.0 license. See [LICENSE](LICENSE) for details.

## Acknowledgements

This dataset is part of the DOR project (<https://github.com/wingter562/DISTINT_open_data>) by Dr. Wentai Wu, Jinan University, with primary contribution by Dr. Shenghai Li, South China University of Technology.

**List of contributors:**

- Wentai Wu, JNU
- Shenghai Li, SCUT
- Kaizhe Song, JNU
- Qinan Wu, JNU
- Yukai Wang, JNU

Project contact: wentaiwu[at]jnu[dot]edu[dot]cn | lishenghai2022[at]foxmail[dot]com

Issues and feature requests: please open a GitHub Issue
