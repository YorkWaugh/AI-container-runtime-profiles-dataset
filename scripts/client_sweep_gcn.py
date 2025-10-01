import time, csv, argparse, requests, torch
import networkx as nx
import random
import numpy as np

def generate_graph_data(num_nodes, num_features, m_edges, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    g = nx.barabasi_albert_graph(n=num_nodes, m=m_edges, seed=seed)
    edge_index = torch.tensor(list(g.edges), dtype=torch.long).t().contiguous()
    x = torch.randn(num_nodes, num_features)
    return x, edge_index

def send_request(url, x_list, edge_index_list):
    t0 = time.perf_counter()
    payload = {"x": x_list, "edge_index": edge_index_list}
    r = requests.post(url, json=payload, timeout=120)
    return time.perf_counter() - t0, r.status_code

def main():
    ap = argparse.ArgumentParser(description="Deterministic GCN Performance Evaluation Client")
    ap.add_argument("--url", required=True, help="URL of the GCN server endpoint.")
    ap.add_argument("--node_scales", nargs="+", type=int, required=True, help="A list of node counts to test.")
    ap.add_argument("--features", type=int, default=16, help="Dimension of node features.")
    ap.add_argument("--m_edges", type=int, default=5, help="Number of edges to attach from a new node to existing nodes in BA model.")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed for deterministic graph generation.")
    ap.add_argument("--out", required=True, help="Output CSV file path.")
    args = ap.parse_args()

    print(f"Starting sweep for URL: {args.url} with global seed {args.seed}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_nodes", "num_edges", "features", "m_param", "seed", "latency_s", "status"])
        for n in args.node_scales:
            try:
                graph_seed = args.seed + n
                x, edge_index = generate_graph_data(n, args.features, args.m_edges, seed=graph_seed)
                num_edges = edge_index.size(1)
                print(f"Testing with {n} nodes, {num_edges} edges (seed={graph_seed})...")
                lat, code = send_request(args.url, x.tolist(), edge_index.tolist())
                w.writerow([n, num_edges, args.features, args.m_edges, graph_seed, lat, code])
                print(f"  -> Latency: {lat:.4f}s, Status: {code}")
            except Exception as e:
                print(f"Error processing {n} nodes: {e}")
                w.writerow([n, -1, args.features, args.m_edges, graph_seed, -1, "error"])

    print(f"Sweep complete. Results saved to {args.out}")

if __name__ == "__main__":
    main()