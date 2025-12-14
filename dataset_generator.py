import json, os
import numpy as np
import networkx as nx


def make_graph(family, n, rng, params):
    if family == "er":
        p = params.get("p", 0.2)
        return nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1e9)))
    
    if family == "ba":
        m = params.get("m", 2)
        return nx.barabasi_albert_graph(n, m, seed=int(rng.integers(1e9)))
    
    if family == "ws":
        k = params.get("k", 4)
        beta = params.get("beta", 0.2)
        return nx.watts_strogatz_graph(n, k, beta, seed=int(rng.integers(1e9)))
    
    if family == "sbm":
        k = params.get("k", 2)
        p_in = params.get("p_in", 0.6)
        p_out = params.get("p_out", 0.05)
        sizes = [n // k] * k
        sizes[-1] += n - sum(sizes)
        P = np.full((k, k), p_out)
        np.fill_diagonal(P, p_in)
        G = nx.stochastic_block_model(sizes, P, seed=int(rng.integers(1e9)))
        return G
    
    if family == "star":
        return nx.star_graph(n - 1)
    
    raise ValueError(f"Unknown family: {family}")

def graph_to_record(G, graph_id, family, n, params, seed):
    edges = [[int(u), int(v)] for u, v in G.edges()]
    deg = [int(G.degree(i)) for i in range(n)]
    return {"graph_id": graph_id, "family": family, "n": n, "params": params, "seed": seed, "edges": edges, "degree": deg,}


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def main(out_dir="data",n=10, n_graphs_per_family=40, train_frac=0.8, seed=0,):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    families = [("er",  {"p": 0.25}), ("ba",  {"m": 2}), ("ws",  {"k": 4, "beta": 0.2}), ("sbm", {"k": 2, "p_in": 0.6, "p_out": 0.05}),("star", {}),]
    graphs = []
    for family, params in families:
        for i in range(n_graphs_per_family):
            gseed = int(rng.integers(1e9))
            Grng = np.random.default_rng(gseed)
            G = make_graph(family, n, Grng, params)
            graph_id = f"{family}_n{n}_seed{gseed}_i{i}"
            graphs.append((graph_id, family, params, gseed, G))

    rng.shuffle(graphs)
    n_train = int(len(graphs) * train_frac)
    train_graphs = graphs[:n_train]
    test_graphs = graphs[n_train:]

    train_graph_records = []
    test_graph_records = []
    for (graph_id, family, params, gseed, G) in train_graphs:
        train_graph_records.append(graph_to_record(G, graph_id, family, n, params, gseed))
    for (graph_id, family, params, gseed, G) in test_graphs:
        test_graph_records.append(graph_to_record(G, graph_id, family, n, params, gseed))

    write_jsonl(os.path.join(out_dir, "graphs_train.jsonl"), train_graph_records)
    write_jsonl(os.path.join(out_dir, "graphs_test.jsonl"), test_graph_records)


    print("Wrote:")
    print("  graphs_train.jsonl, graphs_test.jsonl")

if __name__ == "__main__":
    main()
