# plot_eval_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "eval_metrics_stress.csv" 
OUT_DIR  = "logs"          
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

policy_order = ["uniform", "random", "stage1_only", "stage2"]
df["policy"] = pd.Categorical(df["policy"], categories=policy_order, ordered=True)

def mean_std_bar(metric: str, title: str, fname: str):
    g = df.groupby("policy")[metric].agg(["mean", "std"]).reindex(policy_order)
    x = range(len(g))
    plt.figure()
    plt.bar(x, g["mean"].values, yerr=g["std"].values, capsize=4)
    plt.xticks(x, g.index.tolist(), rotation=0)
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()

mean_std_bar("trust_gain_sum", "Trust gained over episode (mean ± std)", "trust_gain_mean_std.png")
mean_std_bar("return",         "Episode return (mean ± std)",           "return_mean_std.png")
mean_std_bar("penalty_sum",    "Penalty sum (mean ± std)",              "penalty_mean_std.png")
mean_std_bar("shortfall_sum",  "Shortfall sum (mean ± std)",            "shortfall_mean_std.png")
mean_std_bar("oversupply_sum", "Oversupply sum (mean ± std)",           "oversupply_mean_std.png")

plt.figure()
df.boxplot(column="trust_gain_sum", by="policy")
plt.suptitle("")
plt.title("Trust gain distribution by policy")
plt.ylabel("trust_gain_sum")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trust_gain_boxplot.png"), dpi=200)
plt.close()

plt.figure()
for pol in policy_order:
    sub = df[df["policy"] == pol]
    if len(sub) == 0:
        continue
    plt.scatter(sub["trust_gain_sum"], sub["return"], label=pol, alpha=0.7)
plt.xlabel("trust_gain_sum")
plt.ylabel("return")
plt.title("Trust gain vs Return")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trust_vs_return_scatter.png"), dpi=200)
plt.close()

fam = df.groupby(["family", "policy"])["trust_gain_sum"].mean().reset_index()
families = sorted(df["family"].dropna().unique().tolist())

for f in families:
    sub = fam[fam["family"] == f].set_index("policy").reindex(policy_order)
    plt.figure()
    plt.bar(range(len(policy_order)), sub["trust_gain_sum"].values)
    plt.xticks(range(len(policy_order)), policy_order)
    plt.ylabel("mean trust_gain_sum")
    plt.title(f"Mean trust gain by policy — family={f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"trust_gain_by_policy_family_{f}.png"), dpi=200)
    plt.close()

quick = df.groupby("policy")["trust_gain_sum"].mean().reindex(policy_order)
plt.figure()
plt.bar(range(len(policy_order)), quick.values)
plt.xticks(range(len(policy_order)), policy_order)
plt.ylabel("mean trust_gain_sum")
plt.title("Quickest trust proxy (higher = faster trust increase per episode)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "quickest_trust_proxy.png"), dpi=200)
plt.close()

print(f"Saved plots to: {OUT_DIR}/")
print(df.groupby("policy")[["trust_gain_sum","return","penalty_sum","shortfall_sum","oversupply_sum"]].mean().reindex(policy_order))
