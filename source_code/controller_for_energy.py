import argparse
import numpy as np
import csv
import time
import threading
import pynvml
import matplotlib.pyplot as plt
import os

# 导入自定义模块 (确保 client.py 和 processors.py 在同级目录)
from client import AIContainerClient
from processors import ImageProcessor, TextProcessor 

# ================= 1. 优化后的功耗监控类 =================
class GPUEnergyMonitor:
    def __init__(self, device_index=0, interval=0.05):
        self.device_index = device_index
        self.interval = interval
        self.running = False
        self.power_readings = []
        self.timestamps = []  # 记录真实时间戳轴
        self.handle = None
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            print(f"🔌 已连接 GPU: {self.gpu_name} (采样间隔: {self.interval}s)")
        except Exception as e:
            print(f"⚠️ NVML 初始化警告: {e}")

    def get_instant_power(self):
        """获取当前瞬时功率 (W)"""
        if self.handle:
            try:
                return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            except: return 0.0
        return 0.0

    def _record_loop(self):
        start_record_time = time.perf_counter()
        while self.running:
            if self.handle:
                try:
                    p = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    curr_time = time.perf_counter() - start_record_time
                    self.power_readings.append(p)
                    self.timestamps.append(curr_time)
                except:
                    pass
            time.sleep(self.interval)

    def start(self):
        self.power_readings = []
        self.timestamps = []
        self.running = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        
        if len(self.power_readings) < 2:
            return 0.0, 0.0, 0.0

        avg_power = np.mean(self.power_readings)
        peak_power = np.max(self.power_readings)
        
        # 使用梯形积分法算出总能量 (焦耳)
        total_energy_joules = np.trapz(self.power_readings, x=self.timestamps)
        
        return avg_power, peak_power, total_energy_joules

    def cleanup(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass

# ================= 2. 数据保存与绘图逻辑 =================
def save_power_csv(results, filename):
    if not results:
        print("⚠️ 没有结果可以保存")
        return
    
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["scale", "avg_power_active", "peak_power", "energy_per_infer_joules", "latency_per_infer_s"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({
                "scale": f"{res['scale']:.4f}",
                "avg_power_active": f"{res['avg_power_active']:.2f}",
                "peak_power": f"{res['peak_power']:.2f}",
                "energy_per_infer_joules": f"{res['energy_per_infer_joules']:.4f}",
                "latency_per_infer_s": f"{res['latency_per_infer_s']:.4f}"
            })
    print(f"\n💾 结果已保存: {filename}")

def plot_power_results(results):
    if not results: return
    scales = [r["scale"] for r in results]
    energies = [r["energy_per_infer_joules"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(scales, energies, 'o-', color='tab:red', label='Energy per Inference (J)')
    plt.xlabel('Scale Factor')
    plt.ylabel('Energy (Joules)')
    plt.title('GPU Energy Consumption Analysis (Active Only)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

# ================= 3. 核心实验控制逻辑 =================
def run_power_experiment(client, data_path, output_csv):
    monitor = GPUEnergyMonitor(device_index=0, interval=0.05)
    
    print(f"📂 正在加载测试数据: {data_path}")
    raw_data = client.processor.load_data(data_path)

    # A. 测量基准静默功耗 (Idle Power)
    print("💤 正在测量 GPU 静默功耗 (基准去噪)...")
    idle_samples = []
    for _ in range(20):
        idle_samples.append(monitor.get_instant_power())
        time.sleep(0.1)
    avg_idle_power = np.mean(idle_samples)
    print(f"   基准功耗: {avg_idle_power:.2f} W")

    # B. 预热 (Warm-up)
    print(f"🔥 正在预热 (Scale 0.1)...")
    try:
        client.predict(raw_data, scale_factor=0.1)
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ 预热失败: {e}")

    # C. 开始 Scaling 测试
    scales = np.linspace(0.1, 2.0, 20)
    results = []

    print("\n⚡ 开始能耗测试 (Continuous Load Mode)...")
    print("-" * 90)
    print(f"{'Scale':<6} | {'平均功率(W)':<12} | {'单次能耗(J)':<15} | {'单次耗时(s)':<15}")
    print("-" * 90)

    try:
        for s in scales:
            # 动态批处理大小：小图多跑，大图少跑
            BATCH_SIZE = 20 if s < 0.5 else (10 if s < 1.0 else 5)

            monitor.start()
            start_time = time.perf_counter()
            
            success_count = 0
            for _ in range(BATCH_SIZE):
                try:
                    _, latency = client.predict(raw_data, scale_factor=s)
                    if latency > 0: success_count += 1
                except:
                    pass
            
            avg_p_total, peak_p, total_energy_raw = monitor.stop()
            total_duration = time.perf_counter() - start_time

            if success_count > 0:
                # 剥离静默功耗，计算纯推理能效
                # 纯能耗 = (总积分能量 - 静默功率 * 运行时间) / 成功次数
                active_energy_total = total_energy_raw - (avg_idle_power * total_duration)
                energy_per_infer = max(0, active_energy_total / success_count)
                time_per_infer = total_duration / success_count
                
                print(f"{s:<6.2f} | {avg_p_total:<12.2f} | {energy_per_infer:<15.4f} | {time_per_infer:<15.4f}")
                
                results.append({
                    "scale": s,
                    "avg_power_active": avg_p_total,
                    "peak_power": peak_p,
                    "energy_per_infer_joules": energy_per_infer,
                    "latency_per_infer_s": time_per_infer
                })
            
            time.sleep(1.5) # 给 GPU 散热

    finally:
        monitor.cleanup()

    # D. 保存与展示结果
    save_power_csv(results, output_csv)
    plot_power_results(results)

# ================= 4. 入口点 =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 需填写真实参数，包括文件路径和容器端口等
    parser.add_argument("--url", type=str, default="http://localhost:9001/predict")
    parser.add_argument("--input", type=str, 
                        default="/home/lishenghai/桌面/Optimized_Framewok/4.jpg", 
                        help="输入文件的路径")
    parser.add_argument("--type", type=str, default="image", choices=["image", "text"])
    parser.add_argument("--out", type=str, default="energy_results.csv")
    args = parser.parse_args()

    # 自动处理代理环境
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

    # 初始化 Processor
    if args.type == "image":
        my_processor = ImageProcessor()
    else:
        my_processor = TextProcessor()

    # 初始化 Client
    client = AIContainerClient(args.url, processor=my_processor)

    # 运行实验
    run_power_experiment(client, args.input, args.out)