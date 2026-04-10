import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# ============================================================
# CONFIG
# ============================================================
V1_FILE = "granger_results_v1_multilag.csv"
V2_FILE = "granger_results_v2_multilag.csv"

OUTPUT_DIR = "granger_network_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep only edges with enough evidence
MIN_DELTA_F = 0.5
MIN_STRESS_SIG_RATE = 0.30


# ============================================================
# LOAD + SUMMARIZE
# ============================================================
def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    needed = ["pair", "X", "Y", "phase_type", "f_stat", "significant"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df[df["phase_type"].isin(["stress", "rest"])].copy()

    df["f_stat"] = pd.to_numeric(df["f_stat"], errors="coerce")
    df["significant"] = df["significant"].astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(df["significant"]).astype(bool)

    if "best_lag" in df.columns:
        df["best_lag"] = pd.to_numeric(df["best_lag"], errors="coerce")

    return df


def summarize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    # Mean F-stat per pair × phase_type
    f_summary = (
        df.groupby(["pair", "X", "Y", "phase_type"], as_index=False)["f_stat"]
        .mean()
        .pivot(index=["pair", "X", "Y"], columns="phase_type", values="f_stat")
        .reset_index()
        .rename(columns={"rest": "rest_f", "stress": "stress_f"})
    )

    # Significance rate per pair × phase_type
    sig_summary = (
        df.groupby(["pair", "X", "Y", "phase_type"], as_index=False)["significant"]
        .mean()
        .pivot(index=["pair", "X", "Y"], columns="phase_type", values="significant")
        .reset_index()
        .rename(columns={"rest": "rest_sig_rate", "stress": "stress_sig_rate"})
    )

    out = f_summary.merge(sig_summary, on=["pair", "X", "Y"], how="outer")

    # Mean lag if available
    if "best_lag" in df.columns:
        lag_summary = (
            df.groupby(["pair", "X", "Y", "phase_type"], as_index=False)["best_lag"]
            .mean()
            .pivot(index=["pair", "X", "Y"], columns="phase_type", values="best_lag")
            .reset_index()
            .rename(columns={"rest": "rest_lag", "stress": "stress_lag"})
        )
        out = out.merge(lag_summary, on=["pair", "X", "Y"], how="left")
    else:
        out["rest_lag"] = np.nan
        out["stress_lag"] = np.nan

    out["rest_f"] = pd.to_numeric(out["rest_f"], errors="coerce")
    out["stress_f"] = pd.to_numeric(out["stress_f"], errors="coerce")
    out["rest_sig_rate"] = pd.to_numeric(out["rest_sig_rate"], errors="coerce")
    out["stress_sig_rate"] = pd.to_numeric(out["stress_sig_rate"], errors="coerce")

    out["delta_f"] = out["stress_f"] - out["rest_f"]

    return out.sort_values("delta_f", ascending=False).reset_index(drop=True)


# ============================================================
# NETWORK BUILDING
# ============================================================
def build_network(summary_df: pd.DataFrame, mode: str) -> nx.DiGraph:
    """
    mode:
      - 'rest'
      - 'stress'
      - 'delta'
    """
    G = nx.DiGraph()

    nodes = sorted(set(summary_df["X"]).union(set(summary_df["Y"])))
    G.add_nodes_from(nodes)

    if mode == "rest":
        use_df = summary_df[
            (summary_df["rest_sig_rate"] >= MIN_STRESS_SIG_RATE)
        ].copy()
        weight_col = "rest_f"

    elif mode == "stress":
        use_df = summary_df[
            (summary_df["stress_sig_rate"] >= MIN_STRESS_SIG_RATE)
        ].copy()
        weight_col = "stress_f"

    elif mode == "delta":
        use_df = summary_df[
            (summary_df["delta_f"] >= MIN_DELTA_F) &
            (summary_df["stress_sig_rate"] >= MIN_STRESS_SIG_RATE)
        ].copy()
        weight_col = "delta_f"

    else:
        raise ValueError("mode must be 'rest', 'stress', or 'delta'")

    for _, row in use_df.iterrows():
        G.add_edge(
            row["X"], row["Y"],
            weight=float(row[weight_col]),
            rest_f=float(row["rest_f"]) if pd.notna(row["rest_f"]) else np.nan,
            stress_f=float(row["stress_f"]) if pd.notna(row["stress_f"]) else np.nan,
            delta_f=float(row["delta_f"]) if pd.notna(row["delta_f"]) else np.nan,
            rest_sig_rate=float(row["rest_sig_rate"]) if pd.notna(row["rest_sig_rate"]) else np.nan,
            stress_sig_rate=float(row["stress_sig_rate"]) if pd.notna(row["stress_sig_rate"]) else np.nan,
            rest_lag=float(row["rest_lag"]) if pd.notna(row["rest_lag"]) else np.nan,
            stress_lag=float(row["stress_lag"]) if pd.notna(row["stress_lag"]) else np.nan,
        )

    return G


# ============================================================
# METRICS
# ============================================================
def network_metrics(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for node in G.nodes():
        in_edges = list(G.in_edges(node, data=True))
        out_edges = list(G.out_edges(node, data=True))

        in_strength = sum(edge_data.get("weight", 0.0) for _, _, edge_data in in_edges)
        out_strength = sum(edge_data.get("weight", 0.0) for _, _, edge_data in out_edges)

        rows.append({
            "node": node,
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
            "total_degree": G.in_degree(node) + G.out_degree(node),
            "in_strength": in_strength,
            "out_strength": out_strength,
            "total_strength": in_strength + out_strength,
        })

    return pd.DataFrame(rows).sort_values("total_strength", ascending=False).reset_index(drop=True)


def graph_summary(G: nx.DiGraph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": density,
    }


# ============================================================
# PLOTTING
# ============================================================
def draw_graph(G: nx.DiGraph, title: str, outpath: str):
    plt.figure(figsize=(8, 6))
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=1800)
    nx.draw_networkx_labels(G, pos, font_size=10)

    if G.number_of_edges() > 0:
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        max_w = max(weights) if len(weights) > 0 else 1.0
        widths = [1.5 + 4.5 * (w / max_w) for w in weights]

        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowstyle="->",
            arrowsize=18,
            width=widths,
            connectionstyle="arc3,rad=0.08"
        )

        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if "delta" in title.lower():
                edge_labels[(u, v)] = f"{d['delta_f']:.2f}"
            elif "stress" in title.lower():
                edge_labels[(u, v)] = f"{d['stress_f']:.2f}"
            else:
                edge_labels[(u, v)] = f"{d['rest_f']:.2f}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# RUN FOR ONE DATASET
# ============================================================
def run_one_dataset(csv_path: str, label: str):
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")

    df = load_results(csv_path)
    summary = summarize_pairs(df)

    summary_file = os.path.join(OUTPUT_DIR, f"{label.lower()}_pair_summary.csv")
    summary.to_csv(summary_file, index=False)

    print("\nTop 10 stronger-under-stress pairs:")
    show_cols = [
        "pair", "rest_f", "stress_f", "delta_f",
        "rest_sig_rate", "stress_sig_rate",
        "rest_lag", "stress_lag"
    ]
    print(summary[show_cols].head(10).to_string(index=False))

    G_rest = build_network(summary, mode="rest")
    G_stress = build_network(summary, mode="stress")
    G_delta = build_network(summary, mode="delta")

    rest_metrics = network_metrics(G_rest)
    stress_metrics = network_metrics(G_stress)
    delta_metrics = network_metrics(G_delta)

    rest_metrics.to_csv(os.path.join(OUTPUT_DIR, f"{label.lower()}_rest_network_metrics.csv"), index=False)
    stress_metrics.to_csv(os.path.join(OUTPUT_DIR, f"{label.lower()}_stress_network_metrics.csv"), index=False)
    delta_metrics.to_csv(os.path.join(OUTPUT_DIR, f"{label.lower()}_delta_network_metrics.csv"), index=False)

    print("\nRest network summary:")
    print(graph_summary(G_rest))
    print("\nStress network summary:")
    print(graph_summary(G_stress))
    print("\nStronger-under-stress network summary:")
    print(graph_summary(G_delta))

    print("\nTop hubs in stress network:")
    print(stress_metrics.head(5).to_string(index=False))

    print("\nTop hubs in stronger-under-stress network:")
    print(delta_metrics.head(5).to_string(index=False))

    draw_graph(
        G_rest,
        f"{label} - Rest Causality Network",
        os.path.join(OUTPUT_DIR, f"{label.lower()}_rest_network.png")
    )
    draw_graph(
        G_stress,
        f"{label} - Stress Causality Network",
        os.path.join(OUTPUT_DIR, f"{label.lower()}_stress_network.png")
    )
    draw_graph(
        G_delta,
        f"{label} - Stronger Under Stress Network",
        os.path.join(OUTPUT_DIR, f"{label.lower()}_delta_network.png")
    )

    return summary


# ============================================================
# V1 vs V2 PATTERN COMPARISON
# ============================================================
def compare_v1_v2(summary_v1: pd.DataFrame, summary_v2: pd.DataFrame):
    merged = summary_v1.merge(
        summary_v2,
        on=["pair", "X", "Y"],
        suffixes=("_v1", "_v2")
    )

    # same pattern = stronger under stress in both datasets
    merged["same_direction"] = (
        (merged["delta_f_v1"] > 0) &
        (merged["delta_f_v2"] > 0)
    )

    merged["both_strong"] = (
        (merged["delta_f_v1"] >= MIN_DELTA_F) &
        (merged["delta_f_v2"] >= MIN_DELTA_F) &
        (merged["stress_sig_rate_v1"] >= MIN_STRESS_SIG_RATE) &
        (merged["stress_sig_rate_v2"] >= MIN_STRESS_SIG_RATE)
    )

    out_file = os.path.join(OUTPUT_DIR, "v1_v2_pattern_comparison.csv")
    merged.to_csv(out_file, index=False)

    print(f"\n{'='*70}")
    print("V1 vs V2 pattern comparison")
    print(f"{'='*70}")

    replicated = merged[merged["same_direction"]].copy()
    replicated = replicated.sort_values(["delta_f_v1", "delta_f_v2"], ascending=False)

    print("\nTop replicated stronger-under-stress pairs:")
    show_cols = [
        "pair",
        "delta_f_v1", "delta_f_v2",
        "stress_sig_rate_v1", "stress_sig_rate_v2",
        "stress_lag_v1", "stress_lag_v2"
    ]
    print(replicated[show_cols].head(10).to_string(index=False))

    print("\nPairs strong in both datasets:")
    print(merged[merged["both_strong"]][show_cols].to_string(index=False))


# ============================================================
# MAIN
# ============================================================
def main():
    summary_v1 = run_one_dataset(V1_FILE, "V1")
    summary_v2 = run_one_dataset(V2_FILE, "V2")

    compare_v1_v2(summary_v1, summary_v2)

    print(f"\nOutputs saved in: {OUTPUT_DIR}")
    print("\nUse these in your report/poster:")
    print("1. v1_stress_network.png and v1_delta_network.png")
    print("2. v2_stress_network.png and v2_delta_network.png")
    print("3. v1_pair_summary.csv and v2_pair_summary.csv")
    print("4. v1_v2_pattern_comparison.csv")


if __name__ == "__main__":
    main()