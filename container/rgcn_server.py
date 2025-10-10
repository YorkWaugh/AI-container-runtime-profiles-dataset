import torch
from flask import Flask, request, jsonify
from torch_geometric.nn import RGCNConv
import time

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return x

NUM_RELATIONS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RGCN(
    in_channels=16, 
    hidden_channels=32, 
    out_channels=7,
    num_relations=NUM_RELATIONS 
).to(device)
model.eval()
print(f"RGCN Model loaded on {device} with {NUM_RELATIONS} relations")


app = Flask(__name__)

@app.route("/infer", methods=['POST'])
def infer():
    data = request.get_json()

    x = torch.tensor(data['x']).to(device)
    edge_index = torch.tensor(data['edge_index']).to(device)
    edge_type = torch.tensor(data['edge_type']).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(x, edge_index, edge_type)
    latency = time.perf_counter() - t0

    return jsonify({
        "status": "success",
        "latency": latency,
        "output_shape": list(output.shape)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8006, debug=False)