# scripts/client_sweep.py
import time, csv, argparse, requests, numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_image(img, size):
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tfm(img).unsqueeze(0).numpy().tolist()

def send_request(url, tensor, size):
    t0 = time.perf_counter()
    r = requests.post(url, json={"tensor": tensor, "orig_size": [size[1], size[0]]}, timeout=60)
    return time.perf_counter() - t0, r.status_code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--scales", nargs="+", type=float, required=True)
    ap.add_argument("--base_w", type=int, default=1920)
    ap.add_argument("--base_h", type=int, default=1080)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scale","width","height","latency_s","status"])
        for s in args.scales:
            size = (int(args.base_w*s), int(args.base_h*s))
            tensor = preprocess_image(img, size)
            lat, code = send_request(args.url, tensor, size)
            w.writerow([s, size[0], size[1], lat, code])

if __name__ == "__main__":
    main()