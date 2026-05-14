#!/usr/bin/env python3
"""Plot per-epoch loss trajectories from analysis/training_logs.csv.

Usage examples:

  # PPOT: compare a healthy run (RealWorld source) against a collapse case
  # (MHEALTH source) on the OSDA scenario.
  python plot_loss_curves.py --preset ppot_collapse

  # RAINCOAT: visualise the main-phase / correction-phase boundary.
  python plot_loss_curves.py --preset raincoat_phases

  # DANCE: entropy + neighbourhood-clustering losses.
  python plot_loss_curves.py --preset dance_entropy

  # Generic: all metrics for one (scenario, pair, algo, n) cell.
  python plot_loss_curves.py --scenario OSDA --algorithm UniJDOT \
                             --pair "RealWorld -> Pamap2" --n 4

Outputs: figures/loss_<name>.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "analysis", "training_logs.csv")
OUT_DIR  = os.path.join(ROOT, "figures")

PHASE_ORDER = ["pretrain", "main", "align", "correct"]
PHASE_LABEL = {"pretrain": "pretrain", "main": "main", "align": "align",
               "correct": "correct"}


def load():
    df = pd.read_csv(CSV_PATH)
    df["n_unknown"] = pd.to_numeric(df["n_unknown"], errors="coerce")
    df["value"]     = pd.to_numeric(df["value"], errors="coerce")
    df["epoch"]     = pd.to_numeric(df["epoch"], errors="coerce")
    return df


def _to_global_epoch(df_slice):
    """Concatenate phases into one timeline: pretrain (1..P), then
    main/align (1..M -> P+1..P+M), then correct (1..C -> P+M+1..)."""
    out = df_slice.copy()
    phase_max = (df_slice.groupby("phase")["epoch"].max()
                 .reindex(PHASE_ORDER, fill_value=0))
    offset = 0
    offsets = {}
    for ph in PHASE_ORDER:
        if phase_max[ph] > 0:
            offsets[ph] = offset
            offset += int(phase_max[ph])
    out["global_epoch"] = out.apply(
        lambda r: r["epoch"] + offsets.get(r["phase"], 0), axis=1)
    return out, offsets


def _draw_panel(ax, sub, title, share_y=False, log_y=False):
    if sub.empty:
        ax.set_title(f"{title} (no data)", fontsize=10)
        ax.set_axis_off()
        return
    sub, offsets = _to_global_epoch(sub)

    metrics = sub["metric"].unique().tolist()
    cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(metrics))))
    colors = {m: cmap[i % 10] for i, m in enumerate(sorted(metrics))}

    for m in sorted(metrics):
        g = (sub[sub.metric == m]
             .groupby("global_epoch")["value"].agg(["mean", "std"]))
        g = g.sort_index()
        xs = g.index.values.astype(float)
        mu = g["mean"].values
        sd = g["std"].fillna(0.0).values
        ax.plot(xs, mu, label=m, color=colors[m], linewidth=1.4)
        ax.fill_between(xs, mu - sd, mu + sd, color=colors[m], alpha=0.15)

    # Phase boundaries
    cum = 0
    phase_max = sub.groupby("phase")["epoch"].max().reindex(
        PHASE_ORDER, fill_value=0)
    for ph in PHASE_ORDER:
        if phase_max[ph] > 0:
            cum += int(phase_max[ph])
            if ph != PHASE_ORDER[-1]:
                ax.axvline(cum + 0.5, color="gray",
                           linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch (cumulative across phases)")
    ax.set_ylabel("loss value")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def plot_generic(df, scenario, algorithm, pair, n, out_path):
    sub = df[(df.scenario == scenario) & (df.algorithm == algorithm)
             & (df.pair == pair) & (df.n_unknown == n)]
    fig, ax = plt.subplots(figsize=(9, 5))
    title = f"{algorithm}  {pair}  {scenario} n={n}"
    _draw_panel(ax, sub, title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


# ---- presets -----------------------------------------------------------

def preset_ppot_collapse(df, out_path):
    """PPOT on OSDA n=4: compare RealWorld→Pamap2 (works) vs MHEALTH→Pamap2 (collapses)."""
    pairs_to_show = [("RealWorld -> Pamap2", "healthy"),
                     ("MHEALTH -> Pamap2",   "collapse")]
    keep_metrics = {"OT Loss", "Entropic Loss", "Src_cls_loss", "Total_loss"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, (pair, tag) in zip(axes, pairs_to_show):
        sub = df[(df.scenario == "OSDA") & (df.algorithm == "PPOT")
                 & (df.pair == pair) & (df.n_unknown == 4)
                 & (df.metric.isin(keep_metrics))]
        _draw_panel(ax, sub, f"PPOT  {pair}  ({tag})")
    fig.suptitle("PPOT loss trajectories on OSDA n=4 — healthy vs collapse pair",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def preset_raincoat_phases(df, out_path):
    """RAINCOAT on UniDA n=1: show alignment (Sink, Recon) then correction (Recon_correct)."""
    pair = "RealWorld -> Pamap2"
    sub = df[(df.scenario == "UniDA") & (df.algorithm == "RAINCOAT")
             & (df.pair == pair) & (df.n_unknown == 1)]
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_panel(ax, sub, f"RAINCOAT  {pair}  UniDA n=1  (pretrain / align / correct phases)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def preset_dance_entropy(df, out_path):
    """DANCE entropy + neighbourhood losses on RealWorld→Pamap2 (where it partially fires)
    vs MHEALTH→Pamap2 (where it collapses)."""
    pairs_to_show = [("RealWorld -> Pamap2", "partial"),
                     ("MHEALTH -> Pamap2",   "collapse")]
    keep_metrics = {"Ent Loss", "Neighbors Clustering", "Src_cls_loss", "Total_loss"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, (pair, tag) in zip(axes, pairs_to_show):
        sub = df[(df.scenario == "OSDA") & (df.algorithm == "DANCE")
                 & (df.pair == pair) & (df.n_unknown == 4)
                 & (df.metric.isin(keep_metrics))]
        _draw_panel(ax, sub, f"DANCE  {pair}  ({tag})")
    fig.suptitle("DANCE entropy & neighbourhood-clustering losses on OSDA n=4",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def preset_osbp_n_axis(df, out_path):
    """OSBP on OSDA: trajectories at n=1 vs n=8 on RealWorld→Pamap2."""
    pair = "RealWorld -> Pamap2"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, n in zip(axes, [1, 8]):
        sub = df[(df.scenario == "OSDA") & (df.algorithm == "OSBP")
                 & (df.pair == pair) & (df.n_unknown == n)]
        _draw_panel(ax, sub, f"OSBP  {pair}  OSDA n={n}")
    fig.suptitle("OSBP loss trajectories as target-private count grows",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def preset_unijdot_clean(df, out_path):
    """UniJDOT: well-behaved monotone descent example."""
    pair = "RealWorld -> Pamap2"
    sub = df[(df.scenario == "UniDA") & (df.algorithm == "UniJDOT")
             & (df.pair == pair) & (df.n_unknown == 4)]
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_panel(ax, sub, f"UniJDOT  {pair}  UniDA n=4  (clean monotone descent)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


PRESETS = {
    "ppot_collapse":   preset_ppot_collapse,
    "raincoat_phases": preset_raincoat_phases,
    "dance_entropy":   preset_dance_entropy,
    "osbp_n_axis":     preset_osbp_n_axis,
    "unijdot_clean":   preset_unijdot_clean,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(PRESETS))
    p.add_argument("--scenario", choices=["closed_set", "OSDA", "PDA", "UniDA"])
    p.add_argument("--algorithm")
    p.add_argument("--pair")
    p.add_argument("--n", type=int)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()

    if args.preset:
        out = args.out or os.path.join(OUT_DIR, f"loss_{args.preset}.png")
        PRESETS[args.preset](df, out)
    else:
        for k in ("scenario", "algorithm", "pair", "n"):
            if getattr(args, k) is None:
                p.error(f"--{k} is required when --preset is not given")
        out = args.out or os.path.join(
            OUT_DIR,
            f"loss_{args.scenario.lower()}_{args.algorithm}_"
            f"{args.pair.replace(' -> ','_to_').replace(' ','_')}_n{args.n}.png")
        plot_generic(df, args.scenario, args.algorithm, args.pair, args.n, out)


if __name__ == "__main__":
    main()
