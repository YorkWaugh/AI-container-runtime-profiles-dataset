import torch
from torch_geometric.nn import GCNConv

print("Initializing a GCN model to cache any necessary components...")

model = torch.nn.Sequential(
    GCNConv(16, 32),
    torch.nn.ReLU(),
    GCNConv(32, 64)
)
print("GCN components are ready.")