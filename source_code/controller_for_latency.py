import argparse
import numpy as np
import csv
import os
from client import AIContainerClient
# 导入所有可用的处理器
from processors import ImageProcessor, TextProcessor 

def run_experiment(client, data_path, output_csv):
    # 1. 加载数据 (委托给 processor)
    print(f"📂 正在加载数据: {data_path}")
    raw_data = client.processor.load_data(data_path)

    # 2. Warm-up
    print(f"🔥 正在预热 (Target: {client.url})...")
    _, warmup_time = client.predict(raw_data, scale_factor=1.0)
    print(f"   预热耗时: {warmup_time:.4f}s")

    # 3. 定义 Scale 范围
    # 注意：这里的语义取决于 Processor。
    # 对图片是分辨率比例，对文本可能是长度比例。
    scales = np.linspace(0.1, 2.0, 10) 
    results = []

    print("\n🚀 开始 Scaling 测试...")
    for s in scales:
        _, elapsed = client.predict(raw_data, scale_factor=s)
        
        if elapsed > 0:
            print(f"   Scale {s:.2f} | Latency: {elapsed:.4f}s")
            results.append({"scale": s, "time": elapsed})
        else:
            print(f"   Scale {s:.2f} | ❌ Failed")

    # 4. 保存结果
    save_csv(results, output_csv)

def save_csv(results, filename):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["scale", "time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({"scale": f"{res['scale']:.4f}", "time": f"{res['time']:.4f}"})
    print(f"\n💾 结果已保存: {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:9001/predict")
    parser.add_argument("--input", type=str, 
                        default="/home/lishenghai/桌面/Optimized_Framewok/4.jpg", 
                        help="输入图片文件的路径")
    
    parser.add_argument("--type", type=str, default="image", choices=["image", "text"])
    parser.add_argument("--out", type=str, default="result.csv")
    args = parser.parse_args()

    # ================== 核心工厂逻辑 ==================
    # 根据用户输入，组装不同的 Client
    if args.type == "image":
        my_processor = ImageProcessor()
    elif args.type == "text":
        my_processor = TextProcessor()
    else:
        raise ValueError("Unknown type")

    # Client 不知道它是处理图片的，它只知道怎么发包
    client = AIContainerClient(args.url, processor=my_processor)

    run_experiment(client, args.input, args.out)