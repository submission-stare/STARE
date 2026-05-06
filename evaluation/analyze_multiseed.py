"""Analyse multi-seed DRL results and produce paper-ready artifacts.

Reads all aggregated CSVs from the multi_seed/ directory tree, generates:
  - consolidated CSV/JSON across agents & scenarios
  - LaTeX tables (per-metric mean ± std with bootstrap CI)
  - paired statistical tests per agent
  - matplotlib figures (bar plots and box plots)

Run:
    python -m evaluation.analyze_multiseed \\
        --input-dir results/multi_seed \\
        --output-dir results/multi_seed/analysis
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SCENARIOS = ("turb", "noturb", "synth")
AGENTS = ("a2c", "ppo", "ddpg", "td3")

# Metrics shown in the main results table (paper-friendly subset).
PRIMARY_METRICS = ("SR", "Sortino", "AR", "MaxDrawdown", "PSR", "DSR")
PRETTY_NAMES = {
    "SR": "Sharpe",
    "Sortino": "Sortino",
    "AR": "Ann.\\ Return",
    "TotalReturn": "Tot.\\ Return",
    "MaxDrawdown": "Max DD",
    "PSR": "PSR",
    "DSR": "DSR",
    "MinTRL": "MinTRL",
    "FinalValue": "Final Value",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all_runs(input_dir: str) -> pd.DataFrame:
    """Concatenate all per-batch all_runs.csv into a single DataFrame."""
    paths = sorted(glob(os.path.join(input_dir, "*", "all_runs.csv")))
    if not paths:
        raise FileNotFoundError(f"No all_runs.csv files found under {input_dir}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # Drop failed runs (those with no SR)
    df = df.dropna(subset=["SR"])
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10_000, conf: float = 0.95):
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1.0 - conf) / 2.0
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def aggregate(df: pd.DataFrame, metrics=PRIMARY_METRICS) -> pd.DataFrame:
    rows = []
    for (agent, scenario), grp in df.groupby(["agent", "scenario"]):
        row = {"agent": agent, "scenario": scenario, "n_seeds": len(grp)}
        for m in metrics:
            vals = grp[m].astype(float).to_numpy()
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            lo, hi = _bootstrap_ci(vals)
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "agent"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def pairwise_tests(df: pd.DataFrame, metric: str = "SR") -> pd.DataFrame:
    """Per agent, paired Welch t-test and Mann-Whitney U for scenario contrasts."""
    contrasts = [("turb", "noturb"), ("turb", "synth"), ("noturb", "synth")]
    rows = []
    for agent in AGENTS:
        for sa, sb in contrasts:
            a = df[(df["agent"] == agent) & (df["scenario"] == sa)][metric].to_numpy()
            b = df[(df["agent"] == agent) & (df["scenario"] == sb)][metric].to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            # Cohen's d (pooled std)
            sd = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2.0)
            cohen_d = (a.mean() - b.mean()) / sd if sd > 0 else 0.0
            rows.append({
                "agent": agent,
                "metric": metric,
                "contrast": f"{sa} vs {sb}",
                "mean_diff": float(a.mean() - b.mean()),
                "cohen_d": float(cohen_d),
                "welch_t": float(t_stat),
                "welch_p": float(t_p),
                "mw_u": float(u_stat),
                "mw_p": float(u_p),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def _fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def latex_main_table(agg: pd.DataFrame, metrics=PRIMARY_METRICS, decimals: int = 3) -> str:
    """One row per (scenario, agent), columns = metrics with mean ± std."""
    cols = "ll" + "r" * len(metrics)
    header = " & ".join(["Scenario", "Agent"] + [PRETTY_NAMES.get(m, m) for m in metrics])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Multi-seed DRL results (10 seeds, 20{,}000 timesteps). "
        "Mean $\\pm$ standard deviation across seeds for each agent and scenario.}",
        "\\label{tab:multiseed_drl}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    last_scenario = None
    for _, row in agg.sort_values(["scenario", "agent"]).iterrows():
        sc = row["scenario"]
        sc_label = sc if sc != last_scenario else ""
        last_scenario = sc
        cells = [_fmt_pm(row[f"{m}_mean"], row[f"{m}_std"], decimals) for m in metrics]
        lines.append(f"{sc_label} & {row['agent'].upper()} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def latex_ci_table(agg: pd.DataFrame, metric: str = "SR", decimals: int = 3) -> str:
    cols = "ll" + "r" * 3
    header = "Scenario & Agent & Mean & 95\\% CI low & 95\\% CI high"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Bootstrap 95\\% confidence intervals for {PRETTY_NAMES.get(metric, metric)} "
        f"across 10 seeds.}}",
        f"\\label{{tab:multiseed_ci_{metric.lower()}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    last_scenario = None
    for _, row in agg.sort_values(["scenario", "agent"]).iterrows():
        sc = row["scenario"]
        sc_label = sc if sc != last_scenario else ""
        last_scenario = sc
        m = row[f"{metric}_mean"]
        lo = row[f"{metric}_ci_lo"]
        hi = row[f"{metric}_ci_hi"]
        lines.append(
            f"{sc_label} & {row['agent'].upper()} & "
            f"${m:.{decimals}f}$ & ${lo:.{decimals}f}$ & ${hi:.{decimals}f}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def latex_tests_table(tests: pd.DataFrame, decimals: int = 3) -> str:
    cols = "lll" + "r" * 4
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pairwise statistical tests on Sharpe ratios across scenarios "
        "(10 seeds per cell). Welch $t$-test (unequal variances) and Mann--Whitney $U$ "
        "two-sided $p$-values; Cohen's $d$ for effect size.}",
        "\\label{tab:multiseed_tests}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "Agent & Contrast & Metric & $\\Delta$mean & Cohen's $d$ & Welch $p$ & MW $p$ \\\\",
        "\\midrule",
    ]
    for _, r in tests.iterrows():
        lines.append(
            f"{r['agent'].upper()} & {r['contrast']} & {r['metric']} & "
            f"${r['mean_diff']:.{decimals}f}$ & ${r['cohen_d']:.{decimals}f}$ & "
            f"${r['welch_p']:.{decimals}f}$ & ${r['mw_p']:.{decimals}f}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_boxplots(df: pd.DataFrame, output_dir: str, metrics=PRIMARY_METRICS) -> None:
    """One boxplot figure per metric, x = agent, hue = scenario."""
    os.makedirs(output_dir, exist_ok=True)
    for m in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        positions = []
        labels = []
        offset = {"turb": -0.25, "noturb": 0.0, "synth": 0.25}
        colors = {"turb": "#4C72B0", "noturb": "#DD8452", "synth": "#55A868"}
        for i, agent in enumerate(AGENTS):
            for scenario in SCENARIOS:
                vals = df[(df["agent"] == agent) & (df["scenario"] == scenario)][m].to_numpy()
                if len(vals) == 0:
                    continue
                pos = i + offset[scenario]
                bp = ax.boxplot(
                    vals, positions=[pos], widths=0.2, patch_artist=True,
                    boxprops=dict(facecolor=colors[scenario], alpha=0.6),
                    medianprops=dict(color="black"),
                )
                positions.append(pos)
            labels.append(agent.upper())
        ax.set_xticks(range(len(AGENTS)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(PRETTY_NAMES.get(m, m).replace("\\", ""))
        ax.set_title(f"{PRETTY_NAMES.get(m, m).replace(chr(92), '')} — multi-seed distribution")
        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=colors[s], alpha=0.6, label=s) for s in SCENARIOS]
        ax.legend(handles=handles, loc="best", frameon=True)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"boxplot_{m}.pdf")
        fig.savefig(out_path)
        fig.savefig(out_path.replace(".pdf", ".png"), dpi=120)
        plt.close(fig)


def plot_bar_with_ci(agg: pd.DataFrame, output_dir: str, metric: str = "SR") -> None:
    """Bar plot of mean with 95% CI error bars, x=agent, grouped by scenario."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    colors = {"turb": "#4C72B0", "noturb": "#DD8452", "synth": "#55A868"}
    x = np.arange(len(AGENTS))
    for i, scenario in enumerate(SCENARIOS):
        means, los, his = [], [], []
        for agent in AGENTS:
            row = agg[(agg["agent"] == agent) & (agg["scenario"] == scenario)]
            if row.empty:
                means.append(np.nan); los.append(0); his.append(0)
                continue
            m = row[f"{metric}_mean"].iloc[0]
            lo = row[f"{metric}_ci_lo"].iloc[0]
            hi = row[f"{metric}_ci_hi"].iloc[0]
            means.append(m)
            los.append(m - lo)
            his.append(hi - m)
        ax.bar(x + (i - 1) * width, means, width, yerr=[los, his],
               label=scenario, color=colors[scenario], alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in AGENTS])
    ax.set_ylabel(PRETTY_NAMES.get(metric, metric).replace("\\", ""))
    ax.set_title(f"{PRETTY_NAMES.get(metric, metric).replace(chr(92), '')} (mean $\\pm$ 95% bootstrap CI)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"bar_{metric}.pdf")
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=os.path.join(
        _PROJECT_ROOT, "results", "multi_seed"))
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to <input-dir>/analysis")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.input_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading from: {args.input_dir}")
    df = load_all_runs(args.input_dir)
    print(f"Loaded {len(df)} runs across {df['agent'].nunique()} agents and "
          f"{df['scenario'].nunique()} scenarios.")
    df.to_csv(os.path.join(out_dir, "all_runs_consolidated.csv"), index=False)

    agg = aggregate(df)
    agg.to_csv(os.path.join(out_dir, "aggregated_consolidated.csv"), index=False)
    agg.to_json(os.path.join(out_dir, "aggregated_consolidated.json"), orient="records", indent=2)

    # LaTeX tables
    with open(os.path.join(out_dir, "table_main.tex"), "w") as f:
        f.write(latex_main_table(agg))
    for metric in ("SR", "Sortino", "PSR", "AR"):
        with open(os.path.join(out_dir, f"table_ci_{metric.lower()}.tex"), "w") as f:
            f.write(latex_ci_table(agg, metric=metric))

    # Statistical tests
    test_frames = []
    for metric in ("SR", "Sortino", "AR", "MaxDrawdown"):
        test_frames.append(pairwise_tests(df, metric=metric))
    tests_all = pd.concat(test_frames, ignore_index=True)
    tests_all.to_csv(os.path.join(out_dir, "pairwise_tests.csv"), index=False)
    with open(os.path.join(out_dir, "table_pairwise_tests.tex"), "w") as f:
        f.write(latex_tests_table(tests_all))

    # Figures
    fig_dir = os.path.join(out_dir, "figures")
    plot_boxplots(df, fig_dir)
    for metric in PRIMARY_METRICS:
        plot_bar_with_ci(agg, fig_dir, metric=metric)

    # Console summary
    print("\n=== Aggregated summary ===")
    summary_cols = ["agent", "scenario", "n_seeds"] + [
        f"{m}_mean" for m in ("SR", "Sortino", "AR", "MaxDrawdown", "PSR", "DSR")
    ]
    with pd.option_context("display.max_columns", None, "display.width", 160,
                           "display.float_format", "{:.3f}".format):
        print(agg[summary_cols].to_string(index=False))

    print(f"\nOutputs written to: {out_dir}")
    print("  - all_runs_consolidated.csv / .json")
    print("  - aggregated_consolidated.csv / .json")
    print("  - table_main.tex, table_ci_<metric>.tex, table_pairwise_tests.tex")
    print("  - pairwise_tests.csv")
    print(f"  - figures/ (boxplot_<metric>.{{pdf,png}}, bar_<metric>.{{pdf,png}})")


if __name__ == "__main__":
    main()
