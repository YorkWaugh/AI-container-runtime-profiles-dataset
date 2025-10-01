import torch
from flask import Flask, request, jsonify
from torch_geometric.nn import GCNConv
import time

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_channels=16, hidden_channels=32, out_channels=7).to(device)
model.eval()
print(f"GCN Model loaded on {device}")


app = Flask(__name__)

@app.route("/infer", methods=['POST'])
def infer():
    data = request.get_json()

    x = torch.tensor(data['x']).to(device)
    edge_index = torch.tensor(data['edge_index']).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(x, edge_index)
    latency = time.perf_counter() - t0

    return jsonify({
        "status": "success",
        "latency": latency,
        "output_shape": list(output.shape)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8006, debug=False)