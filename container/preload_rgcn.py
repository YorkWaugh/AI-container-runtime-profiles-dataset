import torch
from torch_geometric.nn import RGCNConv

print("Initializing an RGCN model to cache any necessary components...")

NUM_RELATIONS = 4

model = torch.nn.Sequential(
    RGCNConv(16, 32, NUM_RELATIONS),
    torch.nn.ReLU(),
    RGCNConv(32, 64, NUM_RELATIONS)
)
print("RGCN components are ready.")