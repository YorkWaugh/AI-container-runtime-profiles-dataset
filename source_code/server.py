import torch
import numpy as np
import traceback
from flask import Flask, request, jsonify
import os
import torchvision.models as models

app = Flask(__name__)

# ===================== 🔥 加载 FCN-ResNet50 模型 =====================
print("🚀 正在加载 FCN-ResNet50 模型...")
os.environ["TORCH_HOME"] = "/opt/torch_hub/"
torch.hub.set_dir("/opt/torch_hub/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 FCN-ResNet50 分割模型
model = models.segmentation.fcn_resnet50(pretrained=True)
model.to(device).eval()
print("✅ 模型加载完成！")

# ===================== 🔥 API 接口 =====================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📥 收到张量处理请求...")

        # 解析 JSON 数据
        data = request.get_json()
        if data is None or "tensor" not in data:
            print("❌ 请求数据格式错误")
            return jsonify({"error": "请提供 'tensor' 字段，表示输入图像的张量"}), 400

        # 将 JSON 中的 'tensor' 转换为 NumPy 数组，并转为 torch.Tensor
        try:
            input_array = np.array(data["tensor"], dtype=np.float32)
            input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)
        except Exception as e:
            print("❌ 'tensor' 解析失败:", str(e))
            return jsonify({"error": "无效的 'tensor' 格式"}), 400

        # 检查输入张量形状是否为 (1, 3, H, W)，H 和 W 可变
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1 or input_tensor.shape[1] != 3:
            print(f"❌ 输入张量形状错误: {input_tensor.shape}")
            return jsonify({"error": "输入张量形状错误，期望形状为 (1, 3, H, W)"}), 400

        print(f"✅ 接收张量，形状: {input_tensor.shape}")

        # 进行模型推理
        with torch.no_grad():
            # 模型返回一个字典，"out" 是分割结果
            output = model(input_tensor)["out"][0]

        # 生成分割结果（取每个像素的类别索引）
        output_predictions = output.argmax(0).cpu().numpy()
        print("✅ 预测完成，返回结果！")
        return jsonify({"segmentation": output_predictions.tolist()})
    
    except Exception as e:
        print("❌ 服务器处理异常:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ===================== 🚀 启动 Flask 服务器 =====================
if __name__ == '__main__':
    print("🌍 服务器启动中...")
    app.run(host="0.0.0.0", port=8006, threaded=True)
