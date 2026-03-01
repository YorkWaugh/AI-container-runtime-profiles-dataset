import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import abc

# ===================== 基类 (接口定义) =====================
class BaseDataProcessor(abc.ABC):
    """
    抽象基类：定义所有数据处理器必须实现的方法
    """
    @abc.abstractmethod
    def load_data(self, source_path):
        """加载本地数据 (图片路径、文本文件等)"""
        pass

    @abc.abstractmethod
    def prepare_payload(self, raw_data, scale_factor=1.0):
        """
        根据缩放因子处理数据，并返回 API 需要的 JSON payload
        :param scale_factor: 缩放系数 (对于图片是分辨率乘数，对于文本可以是长度截断)
        :return: dict (可以直接传给 requests.json 的字典)
        """
        pass

# ===================== 策略 1: 图像处理 (CV) =====================
class ImageProcessor(BaseDataProcessor):
    def __init__(self):
        # 定义通用的预处理 (Normalize 等)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self, source_path):
        img = cv2.imread(source_path)
        if img is None:
            raise ValueError(f"❌ 无法读取图片: {source_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def prepare_payload(self, raw_data, scale_factor=1.0):
        # 1. Scaling 逻辑：调整分辨率
        orig_h, orig_w = raw_data.shape[0], raw_data.shape[1]
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        
        # 避免缩放至 0
        new_w, new_h = max(1, new_w), max(1, new_h)
        
        img_resized = cv2.resize(raw_data, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Tensor 转换逻辑
        img_pil = Image.fromarray(img_resized)
        input_tensor = self.transform(img_pil)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # 3. 构造 Payload
        return {
            "tensor": input_tensor.cpu().numpy().tolist(),
            "orig_size": [orig_h, orig_w], # 某些模型可能需要知道原图大小
            "meta_scale": scale_factor     # 记录元数据
        }

# ===================== 策略 2: 文本处理 (NLP) - 举例 =====================
class TextProcessor(BaseDataProcessor):
    def load_data(self, source_path):
        # 假设输入是一个 txt 文件
        with open(source_path, 'r', encoding='utf-8') as f:
            return f.read()

    def prepare_payload(self, raw_data, scale_factor=1.0):
        # 1. Scaling 逻辑：截断文本长度
        # 假设 scale_factor=1.0 代表原长，0.5 代表截取一半
        target_len = int(len(raw_data) * scale_factor)
        target_len = max(1, target_len)
        processed_text = raw_data[:target_len]

        # 2. 构造 Payload (假设 NLP 容器接收 {"text": "..."})
        return {
            "text": processed_text,
            "length": target_len
        }