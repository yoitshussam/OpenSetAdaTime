#!/usr/bin/env python3
"""Emit per-scenario LaTeX result tables from analysis/runs.csv.

For each scenario we produce one table per relevant metric:
  closed_set : target_f1
  OSDA       : H_score, OS_star, UNK
  PDA        : target_f1
  UniDA      : H_score, OS_star, UNK

Layout (per table):
  - rows    = algorithms (scenario-native first, then UniDA, then closed-set)
  - columns = the 6 directed pairs + an "Avg" column
  - cells   = mean ± std, averaged across seeds AND n_unknown (for OSDA/PDA/UniDA)
  - bold    = best value per pair (excluding the Avg column)

Output: tables/<scenario>_<metric>.tex
"""
import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(ROOT, "analysis", "runs.csv")
OUT_DIR = os.path.join(ROOT, "tables")

PAIRS = [
    "RealWorld -> Pamap2", "RealWorld -> MHEALTH",
    "Pamap2 -> RealWorld",  "Pamap2 -> MHEALTH",
    "MHEALTH -> RealWorld", "MHEALTH -> Pamap2",
]
PAIR_TEX = {p: p.replace(" -> ", r" $\to$ ").replace("RealWorld", "RW")
                    .replace("Pamap2", "PA").replace("MHEALTH", "MH")
            for p in PAIRS}

UNIDA = ["UDA", "OVANet", "DANCE", "PPOT", "UniOT", "UniJDOT", "RAINCOAT"]
OSDA  = ["OSBP", "TSFA"]
PDA   = ["SPADA", "PDAAN"]
CLOSED = [
    "ACON", "AdvSKM", "CDAN", "CLUDA", "CoDATS", "CoTMix", "DAAN", "DANN",
    "DDC", "Deep_Coral", "DIRT", "DSAN", "HoMM", "MMDA", "SASA",
    "SSSS_TSA", "SWL_Adapt", "uDAR",
]

# Which algorithm list to show in each table, in display order.
ROWS_FOR = {
    "closed_set": sorted(CLOSED) + UNIDA,
    "OSDA":  OSDA + UNIDA,
    "PDA":   PDA  + UNIDA,
    "UniDA": UNIDA,
}

# Which metrics to emit per scenario.
METRICS_FOR = {
    "closed_set": [("target_f1", "Target F1")],
    "OSDA":  [("H_score", "H-score"), ("OS_star", "OS*"), ("UNK", "UNK")],
    "PDA":   [("target_f1", "Target F1")],
    "UniDA": [("H_score", "H-score"), ("OS_star", "OS*"), ("UNK", "UNK")],
}

# Best direction (higher-is-better here for every metric we report).
HIGHER_IS_BETTER = True


def latex_escape(s):
    return s.replace("_", r"\_")


def fmt_cell(mean, std):
    if np.isnan(mean):
        return "--"
    return f"{mean:.3f} \\textpm\\ {std:.3f}"


def build_table(df, scenario, metric_col, metric_label):
    sub = df[df["scenario"] == scenario].dropna(subset=[metric_col])
    if sub.empty:
        return None

    rows = [a for a in ROWS_FOR[scenario] if a in sub["algorithm"].unique()]
    pairs = PAIRS

    # mean & std per (algo, pair), averaging over seeds (and n_unknown for
    # OSDA/PDA/UniDA — closed_set has no n axis).
    grouped = (sub.groupby(["algorithm", "pair"])[metric_col]
                  .agg(["mean", "std"]).reset_index())
    mean = grouped.pivot(index="algorithm", columns="pair", values="mean")
    std  = grouped.pivot(index="algorithm", columns="pair", values="std")

    # Identify best per column
    best_per_pair = {}
    for p in pairs:
        if p not in mean.columns:
            continue
        col_vals = mean[p].dropna()
        if col_vals.empty:
            continue
        best_algo = col_vals.idxmax() if HIGHER_IS_BETTER else col_vals.idxmin()
        best_per_pair[p] = best_algo

    # Build LaTeX
    n_cols = len(pairs) + 2   # algo + pairs + Avg
    col_spec = "l" + "c" * (n_cols - 1)
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + f"{scenario}: {metric_label} (mean $\\pm$ std "
                 + ("across seeds and $n$" if scenario != "closed_set"
                    else "across seeds") + "). Bold = best per pair.}")
    lines.append(r"\label{tab:" + f"{scenario.lower()}_{metric_col}" + "}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header = ["Algorithm"] + [PAIR_TEX[p] for p in pairs] + ["Avg"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for algo in rows:
        cells = [latex_escape(algo)]
        algo_means = []
        for p in pairs:
            if p not in mean.columns or algo not in mean.index:
                cells.append("--")
                continue
            mu = mean.loc[algo, p]
            sd = std.loc[algo, p] if (algo in std.index and p in std.columns) else np.nan
            if np.isnan(mu):
                cells.append("--")
            else:
                algo_means.append(mu)
                cell = fmt_cell(mu, 0.0 if np.isnan(sd) else sd)
                if best_per_pair.get(p) == algo:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
        avg = np.mean(algo_means) if algo_means else np.nan
        cells.append("--" if np.isnan(avg) else f"{avg:.3f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


SCENARIO_HEADER = {
    "closed_set": ("Closed-Set Domain Adaptation",
                   "All closed-set baselines and the UniDA methods run with "
                   "the four shared known classes only (no $n_{\\text{unknown}}$ axis). "
                   "Cells report target F1 (mean $\\pm$ std across seeds)."),
    "OSDA":       ("Open-Set Domain Adaptation (OSDA)",
                   "OSDA-native methods (OSBP, TSFA) and the seven UniDA methods. "
                   "Cells report mean $\\pm$ std across seeds and over all "
                   "$n_{\\text{unknown}}$ values evaluated for that pair."),
    "PDA":        ("Partial Domain Adaptation (PDA)",
                   "PDA-native methods (SPADA, PDAAN) and the seven UniDA methods. "
                   "Cells report target F1 (mean $\\pm$ std across seeds and "
                   "$n_{\\text{unknown}}$)."),
    "UniDA":      ("Universal Domain Adaptation (UniDA)",
                   "The seven UniDA methods; both source-private classes are "
                   "added to the source side AND target-private classes to the "
                   "target side as $n_{\\text{unknown}}$ grows. Cells report mean "
                   "$\\pm$ std across seeds and $n_{\\text{unknown}}$."),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV)
    df["n_unknown"] = pd.to_numeric(df["n_unknown"], errors="coerce")
    for c in ("H_score", "OS_star", "UNK", "target_f1", "source_acc"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    individual = []
    combined_parts = [
        r"% =====================================================",
        r"% Extended Results — append to the end of your main tex",
        r"% =====================================================",
        r"\section{Extended Results}",
        r"\label{sec:extended_results}",
        r"This appendix collects the full per-pair result tables for every "
        r"scenario in our benchmark. The aggregated discussion in the main "
        r"body uses these numbers as its source.",
        r"",
    ]

    for scenario, metrics in METRICS_FOR.items():
        title, blurb = SCENARIO_HEADER[scenario]
        combined_parts.append(r"\subsection{" + title + "}")
        combined_parts.append(r"\label{sec:ext_" + scenario.lower() + "}")
        combined_parts.append(blurb)
        combined_parts.append(r"")
        for metric_col, metric_label in metrics:
            tex = build_table(df, scenario, metric_col, metric_label)
            if tex is None:
                continue
            out = os.path.join(OUT_DIR, f"{scenario.lower()}_{metric_col}.tex")
            with open(out, "w") as f:
                f.write(tex + "\n")
            individual.append(out)
            combined_parts.append(tex)
            combined_parts.append(r"")

    combined_path = os.path.join(OUT_DIR, "extended_results.tex")
    with open(combined_path, "w") as f:
        f.write("\n".join(combined_parts) + "\n")

    for p in individual:
        print(f"wrote: {p}")
    print(f"wrote: {combined_path}")


if __name__ == "__main__":
    main()
