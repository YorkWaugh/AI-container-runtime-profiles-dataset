import torch

print("Downloading and caching DenseNet-121 model...")
torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
print("Model caching complete.")