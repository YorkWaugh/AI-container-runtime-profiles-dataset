import time, csv, argparse, requests, torch

def generate_graph_data(num_nodes, num_features, avg_degree):
    x = torch.randn(num_nodes, num_features)
    num_edges = num_nodes * avg_degree
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    return x.tolist(), edge_index.tolist()

def send_request(url, x, edge_index):
    t0 = time.perf_counter()
    r = requests.post(url, json={"x": x, "edge_index": edge_index}, timeout=60)
    return time.perf_counter() - t0, r.status_code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--node_scales", nargs="+", type=int, required=True, help="List of number of nodes to test.")
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_nodes", "num_edges", "features", "latency_s", "status"])
        for n in args.node_scales:
            x, edge_index = generate_graph_data(n, args.features, args.degree)
            num_edges = len(edge_index[0])
            lat, code = send_request(args.url, x, edge_index)
            w.writerow([n, num_edges, args.features, lat, code])
            print(f"Nodes: {n}, Latency: {lat:.4f}s, Status: {code}")

if __name__ == "__main__":
    main()