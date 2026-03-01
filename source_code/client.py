import requests
import time

class AIContainerClient:
    def __init__(self, url, processor, timeout=60):
        """
        :param url: API 地址
        :param processor: 继承自 BaseDataProcessor 的实例 (i.e., ImageProcessor 或 TextProcessor)
        """
        self.url = url
        self.processor = processor 
        self.timeout = timeout

    def predict(self, raw_data, scale_factor=1.0):
        try:
            payload = self.processor.prepare_payload(raw_data, scale_factor)
        except Exception as e:
            print(f"❌ 数据预处理失败: {e}")
            return None, 0.0

        # 2. 发送请求并计时
        start_time = time.perf_counter()
        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout,proxies={"http": "", "https": ""})
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"❌ HTTP 请求失败: {e}")
            return None, 0.0
        
        latency = time.perf_counter() - start_time
        return response.json(), latency