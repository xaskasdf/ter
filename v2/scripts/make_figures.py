#!/usr/bin/env python3
"""Generate paper figures from tersim simulator + measured GPU CSVs.

Run with the dedicated plot venv:
    scripts/.plotenv/bin/python scripts/make_figures.py

Inputs  : docs/data/*.csv
Outputs : docs/figures/*.png  (150 dpi, paper-ready)
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "docs", "data")
FIGS = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIGS, exist_ok=True)


def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def fig_cache_warming():
    """Cache warming + IPC vs repeated kernel invocations (the smoke-test value:
    simulator captures cache/leakage dynamics without real hardware)."""
    rows = read_csv("iters_sweep.csv")
    iters = [int(r["iters"]) for r in rows]
    hit   = [float(r["l1d_hit_rate"]) * 100 for r in rows]
    ipc   = [float(r["ipc"]) for r in rows]
    pj    = [float(r["asic_pJ_per_tok"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.set_xscale("log", base=2)
    ax1.plot(iters, hit, "o-", color="tab:blue", label="L1D hit rate (%)")
    ax1.plot(iters, [v * 100 for v in ipc], "s-", color="tab:green", label="IPC (%)")
    ax1.set_xlabel("kernel invocations (cache reuse)")
    ax1.set_ylabel("L1D hit rate / IPC (%)")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.plot(iters, pj, "^--", color="tab:red", label="energy (pJ/op @22nm)")
    ax2.set_ylabel("energy pJ/op @22nm (leakage-amortized)")
    ax2.set_yscale("log")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    ax1.set_title("tersim: cache warming + leakage amortization\n(tk_matmul_b_9t, simulated)")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_cache_warming.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_asic_scaling():
    """ASIC node scaling: energy down, freq up across 22→7→3nm."""
    rows = read_csv("asic_nodes.csv")
    nodes = [r["node"] for r in rows]
    pj    = [float(r["pJ_per_tok"]) for r in rows]
    freq  = [float(r["freq_GHz"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    x = range(len(nodes))
    ax1.bar(x, pj, width=0.5, color="tab:red", alpha=0.7, label="energy pJ/tok")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(nodes)
    ax1.set_ylabel("energy pJ/token", color="tab:red")
    ax1.set_xlabel("CMOS node")
    ax2 = ax1.twinx()
    ax2.plot(x, freq, "o-", color="tab:blue", label="freq (GHz)")
    ax2.set_ylabel("frequency (GHz)", color="tab:blue")
    ax1.set_title("tersim ASIC model: node scaling\n(Horowitz 2014 + Stillmaker-Baas 2017)")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_asic_scaling.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_gpu_int4tc():
    """Measured RTX 3090: v11 dp4a vs v13 INT4 TC across Llama 1B shapes."""
    rows = read_csv("gpu_int4tc_vs_v11.csv")
    # rows alternate v11 / v13_int4tc_real per shape.
    shapes, v11, v13 = [], [], []
    for r in rows:
        ms = float(r["ms_median"])
        shp = r["shape"].replace("K=", "").replace("_N=", "x")
        if r["kernel"] == "v11":
            shapes.append(shp); v11.append(ms)
        else:
            v13.append(ms)
    import numpy as np
    x = np.arange(len(shapes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, v11, w, label="v11 dp4a (packed-trit)", color="tab:gray")
    ax.bar(x + w/2, v13, w, label="v13 INT4 TC", color="tab:orange")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(shapes, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ms / forward (log, lower=better)")
    ax.set_title("RTX 3090 measured: INT4 TC wins on large-N shapes\n(Wgate/Wup, lm_head); v11 wins small-N")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_gpu_int4tc.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_hybrid_speedup():
    """Hybrid dispatch: per-shape pick of v11 vs int4tc → net speedup."""
    rows = read_csv("gpu_hybrid.csv")
    data = {}
    for r in rows:
        shp = r["shape"].replace("K=", "").replace("_N=", "x")
        data.setdefault(shp, {})[r["config"]] = float(r["ms_median"])
    shapes = list(data.keys())
    import numpy as np
    x = np.arange(len(shapes))
    w = 0.27
    pv11 = [data[s].get("pure_v11", 0) for s in shapes]
    pv13 = [data[s].get("pure_v13_int4tc", 0) for s in shapes]
    phyb = [data[s].get("hybrid_v13", 0) for s in shapes]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, pv11, w, label="pure v11", color="tab:gray")
    ax.bar(x,     pv13, w, label="pure int4tc", color="tab:orange")
    ax.bar(x + w, phyb, w, label="hybrid (pick per shape)", color="tab:green")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(shapes, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ms / forward (log)")
    ax.set_title("Hybrid dispatch picks the per-shape optimum\n"
                 "Llama 1B fabric: 1.074× over pure v11 (measured)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_hybrid_speedup.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    outs = [
        fig_cache_warming(),
        fig_asic_scaling(),
        fig_gpu_int4tc(),
        fig_hybrid_speedup(),
    ]
    for o in outs:
        print("wrote", o)
