import time, csv, argparse, requests, torch
import networkx as nx

def generate_graph_data(num_nodes, num_features, m_edges):
    g = nx.barabasi_albert_graph(n=num_nodes, m=m_edges)
    edge_index = torch.tensor(list(g.edges), dtype=torch.long).t().contiguous()
    x = torch.randn(num_nodes, num_features)
    return x, edge_index

def send_request(url, x_list, edge_index_list):
    t0 = time.perf_counter()
    payload = {"x": x_list, "edge_index": edge_index_list}
    r = requests.post(url, json=payload, timeout=120)
    return time.perf_counter() - t0, r.status_code

def main():
    ap = argparse.ArgumentParser(description="GCN Performance Evaluation Client with NetworkX")
    ap.add_argument("--url", required=True, help="URL of the GCN server endpoint.")
    ap.add_argument("--node_scales", nargs="+", type=int, required=True, help="A list of node counts to test.")
    ap.add_argument("--features", type=int, default=16, help="Dimension of node features.")
    ap.add_argument("--m_edges", type=int, default=5, help="Number of edges to attach from a new node to existing nodes in BA model.")
    ap.add_argument("--out", required=True, help="Output CSV file path.")
    args = ap.parse_args()

    print(f"Starting sweep for URL: {args.url}")
    print(f"Node scales: {args.node_scales}")
    print(f"BA model parameter m = {args.m_edges}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_nodes", "num_edges", "features", "m_param", "latency_s", "status"])
        for n in args.node_scales:
            try:
                x, edge_index = generate_graph_data(n, args.features, args.m_edges)
                num_edges = edge_index.size(1)
                print(f"Testing with {n} nodes and {num_edges} edges...")
                lat, code = send_request(args.url, x.tolist(), edge_index.tolist())
                w.writerow([n, num_edges, args.features, args.m_edges, lat, code])
                print(f"  -> Latency: {lat:.4f}s, Status: {code}")
            except Exception as e:
                print(f"Error processing {n} nodes: {e}")
                w.writerow([n, -1, args.features, args.m_edges, -1, "error"])

    print(f"Sweep complete. Results saved to {args.out}")

if __name__ == "__main__":
    main()