import argparse
import numpy as np
import csv
import time
import json
import os
from client import AIContainerClient
from processors import ImageProcessor, TextProcessor
from controller_for_energy import GPUEnergyMonitor  # 确保使用的是我们优化过的版本

def run_integrated_experiment(client, data_path, output_csv, repeats=5):
    """
    集成的实验控制器
    :param repeats: 每个 Scale 重复实验的次数，用于计算均值和方差
    """
    monitor = GPUEnergyMonitor(device_index=0, interval=0.05)
    
    # 1. 环境准备与数据加载
    print(f"📂 加载测试数据: {data_path}")
    raw_data = client.processor.load_data(data_path)
    
    # 2. 测量环境基准 (Idle Power)
    print("💤 正在测量 GPU 静默功耗 (基准去噪)...")
    idle_samples = [monitor.get_instant_power() for _ in range(20)]
    avg_idle_p = np.mean(idle_samples)
    print(f"   基准功率: {avg_idle_p:.2f} W")

    # 3. 预热 (Warm-up)
    print(f"🔥 正在预热 (Target: {client.url})...")
    client.predict(raw_data, scale_factor=0.1)
    time.sleep(2) # 给 GPU 降温

    # 4. 定义 Scaling 范围
    scales = np.linspace(0.1, 2.0, 20)
    final_results = []

    print(f"\n🚀 开始集成测试 (每组重复 {repeats} 次)...")
    print("-" * 110)
    print(f"{'Scale':<6} | {'Avg Latency(s)':<15} | {'Std Dev':<10} | {'Active Energy(J)':<16} | {'Avg Power(W)':<12}")
    print("-" * 110)

    try:
        for s in scales:
            batch_latencies = []
            
            # 开启能耗监控 (包裹整个重复测试组)
            monitor.start()
            start_wall_clock = time.perf_counter()
            
            # --- 核心循环: 多次重复采样 ---
            for i in range(repeats):
                # 调用我们优化过的 predict (内含预序列化)
                _, latency = client.predict(raw_data, scale_factor=s)
                if latency > 0:
                    batch_latencies.append(latency)
            
            # 停止监控
            avg_p_total, peak_p, total_energy_raw = monitor.stop()
            total_duration = time.perf_counter() - start_wall_clock
            
            if len(batch_latencies) > 0:
                # --- 统计计算 ---
                mean_latency = np.mean(batch_latencies)
                std_latency = np.std(batch_latencies)
                
                # 剥离静默功耗，计算纯推理能耗
                # 纯能效 = (总面积 - 静默面积) / 成功次数
                active_energy_total = total_energy_raw - (avg_idle_p * total_duration)
                energy_per_infer = max(0, active_energy_total / len(batch_latencies))
                
                print(f"{s:<6.2f} | {mean_latency:<15.4f} | {std_latency:<10.4f} | {energy_per_infer:<16.4f} | {avg_p_total:<12.2f}")
                
                final_results.append({
                    "scale": s,
                    "latency_mean": mean_latency,
                    "latency_std": std_latency,
                    "energy_joules": energy_per_infer,
                    "avg_power_watts": avg_p_total,
                    "idle_power_baseline": avg_idle_p
                })

            time.sleep(1.5) # 组间散热

    finally:
        monitor.cleanup()

    save_results(final_results, output_csv)

def save_results(results, filename):
    if not results: return
    with open(filename, "w", newline="") as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in res.items()})
    print(f"\n💾 实验数据已完整保存至: {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AC-Prof Integrated Controller")
    parser.add_argument("--url", type=str, default="http://localhost:8006/predict")
    parser.add_argument("--input", type=str, required=True, help="Path to input data")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repetitions per scale")
    parser.add_argument("--out", type=str, default="ac_prof_results.csv")
    args = parser.parse_args()

    # 环境补丁: 彻底禁用代理，防止干扰本地时延
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

    # 初始化 (默认使用 ImageProcessor)
    client = AIContainerClient(args.url, processor=ImageProcessor())
    
    run_integrated_experiment(client, args.input, args.out, repeats=args.repeats)