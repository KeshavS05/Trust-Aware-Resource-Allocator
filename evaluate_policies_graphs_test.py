import argparse
import json
from dataclasses import dataclass
import networkx as nx
import numpy as np
import pandas as pd

from Step7_Ten_Nodes import ResourceAllocTrustEnv, DDPGAgent, GraphSampler, project_to_simplex


def load_graph_samples(jsonl_path):
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    assert len(samples) > 0
    return samples


def build_nx_graph_from_edges(n: int, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(u), int(v)) for (u, v) in edges)
    return G


class SimpleGraphCycler:
    def __init__(self, jsonl_path, seed = 0, shuffle = True):
        self.samples = load_graph_samples(jsonl_path)
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.samples))
        if shuffle:
            self.rng.shuffle(self.order)
        self.ptr = 0
        self.shuffle = shuffle

    def sample(self):
        idx = int(self.order[self.ptr])
        self.ptr = (self.ptr + 1) % len(self.samples)
        if self.ptr == 0 and self.shuffle:
            self.rng.shuffle(self.order)

        s = self.samples[idx]
        n = int(s["n"])
        G = build_nx_graph_from_edges(n, s["edges"])
        degs = np.array(s.get("degree", [G.degree(i) for i in range(n)]), dtype=np.float32)
        degs_norm = degs / (degs.max() + 1e-8)

        meta = dict(s)
        return G, degs_norm, meta




# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

# Baseline policies
def policy_uniform(n):
    return np.ones(n, dtype=np.float32) / float(n)


def policy_random(n, rng):
    a = rng.random(n, dtype=np.float32)
    return project_to_simplex(a)




# -------------------------
# -------------------------
@dataclass
class AgentPolicy:
    agent: DDPGAgent
    mode: str

    def act(self, state):
        if self.mode == "stage2":
            return self.agent.select_action(state, noise_scale=0.0)

        if self.mode == "stage1_only":
            s = state.copy()
            n = self.agent.n
            s[n:3 * n] = 0.0
            return self.agent.select_action(s, noise_scale=0.0)

        raise ValueError(f"Unknown mode: {self.mode}")





# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

# metrics
def run_episode(env, action_fn, n):
    s = env.reset()
    done = False

    ep_return = 0.0
    trust_gain_sum = 0.0
    penalty_sum = 0.0
    oversupply_sum = 0.0
    shortfall_sum = 0.0
    wants_sum_start = float(np.sum(s[:n]))

    while not done:
        a = action_fn(s)
        ns, r, done, info = env.step(a)
        ep_return += float(r)

        if isinstance(info, dict):
            if "trust_t" in info and "trust_next" in info:
                trust_gain_sum += float(np.sum(info["trust_next"] - info["trust_t"]))
            if "penalty_short" in info:
                penalty_sum += float(info["penalty_short"])
            if "oversupply" in info:
                oversupply_sum += float(np.sum(info["oversupply"]))
            if "shortfall" in info:
                shortfall_sum += float(np.sum(info["shortfall"]))

        s = ns

    wants_sum_end = float(np.sum(s[:n]))
    return {
        "return": ep_return,
        "trust_gain_sum": trust_gain_sum,
        "penalty_sum": penalty_sum,
        "oversupply_sum": oversupply_sum,
        "shortfall_sum": shortfall_sum,
        "wants_sum_start": wants_sum_start,
        "wants_sum_end": wants_sum_end,
    }


def make_env(*, n_nodes, episode_length, seed, alpha, max_want, graph, degs_norm, meta, wants0, trust0):
    env = ResourceAllocTrustEnv(n_nodes=n_nodes, episode_length=episode_length, seed=seed, alpha=alpha, max_want=max_want, wants_init=wants0, trust_init=trust0, graph=graph, graph_sampler=None, resample_graph_each_reset=False,)
    env.degree_vec = degs_norm.astype(np.float32)
    env.graph_meta = meta
    return env


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--graphs_jsonl", type=str, default="data/graphs_test.jsonl")
    ap.add_argument("--episodes", type=int, default=100, help="How many evaluation episodes to run.")
    ap.add_argument("--sample_k_graphs", type=int, default=-1, help="If >0, sample k graphs from file and cycle them. Else use all.")

    ap.add_argument("--stage2_ckpt", type=str, default="checkpoints/stage2_final.pt")
    ap.add_argument("--stage1_ckpt", type=str, default="checkpoints/stage1_final.pt")

    ap.add_argument("--n_nodes", type=int, default=10)
    ap.add_argument("--episode_length", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--max_want", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_csv", type=str, default="eval_metrics_stress.csv")

    ap.add_argument("--fixed_wants", action="store_true", help="Use a fixed wants0 across all episodes (same for all policies).")
    ap.add_argument("--fixed_trust", action="store_true", help="Use a fixed trust0 across all episodes (same for all policies).")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- Load Stage 2 agent
    stage2_agent = DDPGAgent(state_dim=3 * args.n_nodes, n_nodes=args.n_nodes, device="cpu")
    stage2_agent.load(args.stage2_ckpt, load_optim=False)
    stage2_agent.actor.eval()
    stage2_policy = AgentPolicy(stage2_agent, mode="stage2")

    # ---- Load Stage 1 agent (for stage1-only policy)
    stage1_agent = DDPGAgent(state_dim=3 * args.n_nodes, n_nodes=args.n_nodes, device="cpu")
    stage1_agent.load(args.stage1_ckpt, load_optim=False)
    stage1_agent.actor.eval()
    stage1_only_policy = AgentPolicy(stage1_agent, mode="stage1_only")

    # ---- Graph sampler
    sampler = SimpleGraphCycler(args.graphs_jsonl, seed=args.seed, shuffle=True)
    n_graphs_total = len(sampler.samples)

    if args.sample_k_graphs and args.sample_k_graphs > 0:
        k = min(args.sample_k_graphs, n_graphs_total)
        chosen = rng.choice(n_graphs_total, size=k, replace=False)
        sampler.order = np.array(chosen, dtype=int)
        sampler.ptr = 0
        sampler.shuffle = False

    # ---- Policies to evaluate
    def act_uniform(state): 
        return policy_uniform(args.n_nodes)
    
    def act_random(state):  
        return policy_random(args.n_nodes, rng)
    
    def act_stage2(state):  
        return stage2_policy.act(state)
    
    def act_stage1only(state): 
        return stage1_only_policy.act(state)

    policies = {"uniform": act_uniform, "random": act_random, "stage1_only": act_stage1only, "stage2": act_stage2}

    fixed_wants0 = rng.uniform(0.0, args.max_want, size=args.n_nodes).astype(np.float32) if args.fixed_wants else None
    fixed_trust0 = rng.beta(2.0, 5.0, size=args.n_nodes).astype(np.float32) if args.fixed_trust else None

    rows = []
    for ep_idx in range(args.episodes):
        G, degs_norm, meta = sampler.sample()

        last_idx = int(sampler.order[(sampler.ptr - 1) % len(sampler.order)])
        record = sampler.samples[last_idx]

        if "wants0" in record and record["wants0"] is not None:
            wants0 = np.array(record["wants0"], dtype=np.float32)
        else:
            wants0 = fixed_wants0 if fixed_wants0 is not None else rng.uniform(0.0, args.max_want, size=args.n_nodes).astype(np.float32)
            
        trust0 = fixed_trust0 if fixed_trust0 is not None else rng.beta(2.0, 5.0, size=args.n_nodes).astype(np.float32)

        for pname, action_fn in policies.items():
            env = make_env(
                n_nodes=args.n_nodes,
                episode_length=args.episode_length,
                seed=args.seed + ep_idx,
                alpha=args.alpha,
                max_want=args.max_want,
                graph=G,
                degs_norm=degs_norm,
                meta=meta,
                wants0=wants0,
                trust0=trust0,
            )
            m = run_episode(env, action_fn, n=args.n_nodes)
            rows.append({"episode": ep_idx, "policy": pname, "graph_id": meta.get("graph_id"), "family": meta.get("family"), **m})

        if (ep_idx + 1) % max(1, args.episodes // 10) == 0:
            print(f"[eval] {ep_idx+1}/{args.episodes} episodes done")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
