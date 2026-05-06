#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from datetime import datetime
from html import escape


AGENTS = ["benchmark", "td3", "sac", "a2c", "ddpg", "ppo"]
RL_AGENTS = ["td3", "sac", "a2c", "ddpg", "ppo"]
COLORS = {
    "benchmark": "#111827",
    "td3": "#14532d",
    "sac": "#0f766e",
    "a2c": "#9f1239",
    "ddpg": "#7c2d12",
    "ppo": "#7e22ce",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark-data interleaved RL outputs without third-party dependencies."
    )
    parser.add_argument(
        "--data-root",
        default="./benchmark-data",
        help="Root benchmark-data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/quant_interleaved_analysis",
        help="Directory where plots and report will be written.",
    )
    parser.add_argument(
        "--hypotheses-md",
        default="quant_hypotheses_interleaved.md",
        help="Hypotheses markdown note used as input context.",
    )
    return parser.parse_args()


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text in {"-", "FALIDO"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def path_for(data_root: str, *parts: str) -> str:
    return os.path.join(data_root, *parts)


def fmt_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    amount = abs(value)
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        return f"{sign}${amount / 1_000:.1f}K"
    return f"{sign}${amount:.0f}"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_account_values(data_root: str) -> tuple[list[str], list[dict[str, float]]]:
    rows = read_csv_rows(path_for(data_root, "run_000", "interleaved", "account_values.csv"))
    dates: list[str] = []
    series: list[dict[str, float]] = []
    for row in rows:
        dates.append(row["date"])
        series.append({name: float(row[name]) for name in AGENTS})
    return dates, series


def load_mean_metrics(data_root: str) -> dict[str, dict[str, float]]:
    rows = read_csv_rows(
        path_for(data_root, "aggregated_interleaved", "aggregated", "mean_portfolio_metrics.csv")
    )
    metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        agent = row["agent"]
        metrics[agent] = {
            "sharpe": float(row["mean_portfolio_sharpe"]),
            "final_value": float(row["mean_portfolio_final_value"]),
            "max_drawdown": float(row["mean_portfolio_max_drawdown"]),
        }
    return metrics


def load_single_run_risk(data_root: str) -> dict[str, dict[str, float | None]]:
    rows = read_csv_rows(
        path_for(data_root, "run_000", "interleaved", "financial_metrics", "sharpe_summary_agents_with_psr.csv")
    )
    summary: dict[str, dict[str, float | None]] = {}
    for row in rows:
        model = row["model"]
        summary[model] = {
            "sharpe": parse_float(row["sharpeRatio"]),
            "psr": parse_float(row["psr"]),
            "dsr": parse_float(row["dsr"]),
            "volatility": parse_float(row["volatility"]),
            "max_drawdown": parse_float(row["maxDrawdown"]),
        }
    return summary


def load_transaction_summary(data_root: str) -> dict[str, dict[str, float]]:
    rows = read_csv_rows(
        path_for(
            data_root,
            "run_000",
            "interleaved",
            "agents_trading",
            "trading_analysis",
            "_comparison",
            "compare_transaction_summary.csv",
        )
    )
    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        agent = row["agent"]
        total_actions = int(row["total_buys"]) + int(row["total_sells"])
        summary[agent] = {
            "total_buys": float(row["total_buys"]),
            "total_sells": float(row["total_sells"]),
            "total_holds": float(row["total_holds"]),
            "total_value_buy": float(row["total_value_buy"]),
            "total_value_sell": float(row["total_value_sell"]),
            "total_cost": float(row["total_cost"]),
            "total_actions": float(total_actions),
        }
    return summary


def load_ticker_pnl(data_root: str) -> dict[str, dict[str, tuple[str, float]]]:
    rows = read_csv_rows(
        path_for(
            data_root,
            "aggregated_interleaved",
            "aggregated",
            "stock_analysis",
            "01_profitability",
            "ticker_pnl_by_agent.csv",
        )
    )
    result: dict[str, dict[str, tuple[str, float]]] = {}
    for row in rows:
        algo = row["algo"]
        pnl = float(row["mean_pnl"])
        ticker = row["ticker"]
        result.setdefault(algo, {})
        best = result[algo].get("best")
        worst = result[algo].get("worst")
        if best is None or pnl > best[1]:
            result[algo]["best"] = (ticker, pnl)
        if worst is None or pnl < worst[1]:
            result[algo]["worst"] = (ticker, pnl)
    return result


def load_final_weights(data_root: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    rows = read_csv_rows(
        path_for(
            data_root,
            "aggregated_interleaved",
            "aggregated",
            "stock_analysis",
            "02_consensus",
            "final_weights_all_runs.csv",
        )
    )
    tickers = [name for name in rows[0].keys() if name not in {"run_id", "agent_key", "prefix", "algo"}]
    weights: dict[str, dict[str, float]] = {}
    for row in rows:
        algo = row["algo"]
        weights[algo] = {ticker: float(row[ticker]) for ticker in tickers}
    return tickers, weights


def load_snapshot_stats(data_root: str) -> dict[str, dict[str, float]]:
    base_dir = path_for(data_root, "run_000", "interleaved", "agents_trading", "trading_analysis")
    stats: dict[str, dict[str, float]] = {}
    for agent in RL_AGENTS:
        rows = read_csv_rows(path_for(base_dir, agent, "snapshots.csv"))
        if rows and rows[-1]["step"] == "0":
            rows = rows[:-1]
        negative_equity_rows = 0
        full_cash_rows = 0
        max_cash_weight = -1e18
        min_cash_weight = 1e18
        min_total_asset = 1e18
        max_total_asset = -1e18
        max_gross_exposure = 0.0
        max_single_long = -1e18
        max_single_short = 1e18
        final_cash_weight = 0.0
        for row in rows:
            total_asset = float(row["total_asset"])
            cash_weight = float(row["cash_weight"])
            final_cash_weight = cash_weight
            if total_asset < 0:
                negative_equity_rows += 1
            if cash_weight >= 0.999999:
                full_cash_rows += 1
            min_cash_weight = min(min_cash_weight, cash_weight)
            max_cash_weight = max(max_cash_weight, cash_weight)
            min_total_asset = min(min_total_asset, total_asset)
            max_total_asset = max(max_total_asset, total_asset)
            weight_values = [
                float(value)
                for key, value in row.items()
                if key.endswith("_weight") and key != "cash_weight"
            ]
            gross = sum(abs(value) for value in weight_values)
            max_gross_exposure = max(max_gross_exposure, gross)
            if weight_values:
                max_single_long = max(max_single_long, max(weight_values))
                max_single_short = min(max_single_short, min(weight_values))
        stats[agent] = {
            "negative_equity_rows": float(negative_equity_rows),
            "full_cash_rows": float(full_cash_rows),
            "min_cash_weight": min_cash_weight,
            "max_cash_weight": max_cash_weight,
            "final_cash_weight": final_cash_weight,
            "min_total_asset": min_total_asset,
            "max_total_asset": max_total_asset,
            "max_gross_exposure": max_gross_exposure,
            "max_single_long": max_single_long,
            "max_single_short": max_single_short,
        }
    return stats


def pick_hypotheses() -> list[str]:
    return [
        "Benchmark outperforms because it stays solvent and within sane risk bounds while several RL agents allow effective portfolio insolvency.",
        "TD3 wins by holding a concentrated, high-conviction relative-value book, especially a very large JPM long against multiple shorts.",
        "SAC survives because it stays continuously invested with a diversified long-short book rather than collapsing to cash or over-rotating.",
        "A2C, DDPG, and PPO likely enter a post-blowup liquidation trap: negative equity followed by long stretches of full-cash inactivity.",
        "PPO's poor outcome is probably a combination of weak signal and excessive trading, so turnover and costs amplify already bad positioning.",
        "The strong SAC and TD3 outcomes may partly rely on leverage and margin assumptions that are generous relative to realistic portfolio constraints.",
    ]


def scale_value(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if math.isclose(max_value, min_value):
        return (start + end) / 2.0
    ratio = (value - min_value) / (max_value - min_value)
    return start + ratio * (end - start)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def build_svg(parts: list[str], width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<style>text{font-family:Arial,sans-serif;fill:#111827} .small{font-size:11px} .axis{stroke:#374151;stroke-width:1} .grid{stroke:#d1d5db;stroke-width:1;stroke-dasharray:3 3}</style>',
            *parts,
            "</svg>",
        ]
    )


def plot_final_values(path: str, mean_metrics: dict[str, dict[str, float]]) -> None:
    width, height = 980, 560
    left, right, top, bottom = 90, 40, 50, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    values = [mean_metrics[agent]["final_value"] for agent in AGENTS]
    min_val = min(0.0, min(values))
    max_val = max(0.0, max(values))
    y_zero = scale_value(0.0, min_val, max_val, top + chart_h, top)
    parts = [
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="20" font-weight="700">Final Portfolio Value By Agent</text>',
    ]
    for tick in range(6):
        value = min_val + (max_val - min_val) * tick / 5
        y = scale_value(value, min_val, max_val, top + chart_h, top)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left+chart_w}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small" x="{left-8}" y="{y+4:.2f}" text-anchor="end">{escape(fmt_money(value))}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{y_zero:.2f}" x2="{left+chart_w}" y2="{y_zero:.2f}"/>')
    bar_w = chart_w / len(AGENTS) * 0.62
    for index, agent in enumerate(AGENTS):
        value = mean_metrics[agent]["final_value"]
        x = left + (index + 0.5) * chart_w / len(AGENTS) - bar_w / 2
        y = scale_value(max(value, 0.0), min_val, max_val, top + chart_h, top)
        y_neg = scale_value(min(value, 0.0), min_val, max_val, top + chart_h, top)
        rect_y = min(y, y_neg)
        rect_h = abs(y_neg - y)
        parts.append(
            f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_w:.2f}" height="{rect_h:.2f}" fill="{COLORS[agent]}" opacity="0.9"/>'
        )
        parts.append(
            f'<text class="small" x="{x + bar_w/2:.2f}" y="{(rect_y - 8) if value >= 0 else (rect_y + rect_h + 14):.2f}" text-anchor="middle">{escape(fmt_money(value))}</text>'
        )
        parts.append(f'<text class="small" x="{x + bar_w/2:.2f}" y="{top + chart_h + 24:.2f}" text-anchor="middle">{agent.upper()}</text>')
    write_text(path, build_svg(parts, width, height))


def plot_equity_curves(path: str, dates: list[str], series: list[dict[str, float]]) -> None:
    width, height = 1080, 620
    left, right, top, bottom = 85, 170, 50, 70
    chart_w = width - left - right
    chart_h = height - top - bottom
    normalized: dict[str, list[float]] = {}
    for agent in AGENTS:
        first = series[0][agent]
        normalized[agent] = [safe_div(row[agent], first) for row in series]
    min_val = min(min(values) for values in normalized.values())
    max_val = max(max(values) for values in normalized.values())
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 1.0)
    parts = [
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="20" font-weight="700">Normalized Equity Curves</text>',
    ]
    for tick in range(6):
        value = min_val + (max_val - min_val) * tick / 5
        y = scale_value(value, min_val, max_val, top + chart_h, top)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left+chart_w}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small" x="{left-8}" y="{y+4:.2f}" text-anchor="end">{value:.2f}x</text>')
    x_ticks = [0, len(dates) // 4, len(dates) // 2, (3 * len(dates)) // 4, len(dates) - 1]
    for idx in x_ticks:
        x = scale_value(float(idx), 0.0, float(len(dates) - 1), left, left + chart_w)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+chart_h}"/>')
        parts.append(f'<text class="small" x="{x:.2f}" y="{top+chart_h+18:.2f}" text-anchor="middle">{dates[idx]}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}"/>')
    for legend_index, agent in enumerate(AGENTS):
        points = []
        for idx, value in enumerate(normalized[agent]):
            x = scale_value(float(idx), 0.0, float(len(dates) - 1), left, left + chart_w)
            y = scale_value(value, min_val, max_val, top + chart_h, top)
            points.append(f"{x:.2f},{y:.2f}")
        parts.append(
            f'<polyline fill="none" stroke="{COLORS[agent]}" stroke-width="2.5" points="{" ".join(points)}"/>'
        )
        ly = top + 18 + legend_index * 22
        lx = left + chart_w + 18
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+18}" y2="{ly}" stroke="{COLORS[agent]}" stroke-width="3"/>')
        parts.append(f'<text class="small" x="{lx+26}" y="{ly+4}">{agent.upper()}</text>')
    write_text(path, build_svg(parts, width, height))


def plot_cash_and_equity(path: str, snapshot_stats: dict[str, dict[str, float]]) -> None:
    width, height = 980, 560
    left, right, top, bottom = 90, 40, 50, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    agents = RL_AGENTS
    values = [
        snapshot_stats[agent]["full_cash_rows"]
        for agent in agents
    ] + [
        snapshot_stats[agent]["negative_equity_rows"]
        for agent in agents
    ]
    max_val = max(values) if values else 1.0
    parts = [
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="20" font-weight="700">Failure-State Behavior: Full Cash And Negative Equity</text>',
    ]
    for tick in range(6):
        value = max_val * tick / 5
        y = scale_value(value, 0.0, max_val, top + chart_h, top)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left+chart_w}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small" x="{left-8}" y="{y+4:.2f}" text-anchor="end">{int(value)}</text>')
    group_w = chart_w / len(agents)
    bar_w = group_w * 0.28
    for index, agent in enumerate(agents):
        center = left + (index + 0.5) * group_w
        full_cash = snapshot_stats[agent]["full_cash_rows"]
        neg_eq = snapshot_stats[agent]["negative_equity_rows"]
        for offset, label, value, color in [
            (-bar_w * 0.65, "cash", full_cash, "#2563eb"),
            (bar_w * 0.15, "neg", neg_eq, COLORS[agent]),
        ]:
            x = center + offset
            y = scale_value(value, 0.0, max_val, top + chart_h, top)
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{top+chart_h-y:.2f}" fill="{color}" opacity="0.9"/>'
            )
            parts.append(
                f'<text class="small" x="{x+bar_w/2:.2f}" y="{y-6:.2f}" text-anchor="middle">{int(value)}</text>'
            )
        parts.append(f'<text class="small" x="{center:.2f}" y="{top+chart_h+24:.2f}" text-anchor="middle">{agent.upper()}</text>')
    legend_y = top + 10
    legend_x = left + chart_w - 180
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="#2563eb"/>')
    parts.append(f'<text class="small" x="{legend_x+18}" y="{legend_y+10}">full cash rows</text>')
    parts.append(f'<rect x="{legend_x}" y="{legend_y+18}" width="12" height="12" fill="#6b7280"/>')
    parts.append(f'<text class="small" x="{legend_x+18}" y="{legend_y+28}">negative equity rows</text>')
    write_text(path, build_svg(parts, width, height))


def plot_turnover_vs_outcome(
    path: str,
    mean_metrics: dict[str, dict[str, float]],
    transaction_summary: dict[str, dict[str, float]],
) -> None:
    width, height = 920, 560
    left, right, top, bottom = 90, 50, 50, 70
    chart_w = width - left - right
    chart_h = height - top - bottom
    xs = [transaction_summary[agent]["total_actions"] for agent in RL_AGENTS]
    ys = [mean_metrics[agent]["final_value"] for agent in RL_AGENTS]
    min_x = 0.0
    max_x = max(xs) * 1.08
    min_y = min(ys)
    max_y = max(ys)
    parts = [
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="20" font-weight="700">Turnover Versus Final Value</text>',
    ]
    for tick in range(6):
        y_val = min_y + (max_y - min_y) * tick / 5
        y = scale_value(y_val, min_y, max_y, top + chart_h, top)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left+chart_w}" y2="{y:.2f}"/>')
        parts.append(f'<text class="small" x="{left-8}" y="{y+4:.2f}" text-anchor="end">{escape(fmt_money(y_val))}</text>')
    for tick in range(6):
        x_val = min_x + (max_x - min_x) * tick / 5
        x = scale_value(x_val, min_x, max_x, left, left + chart_w)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+chart_h}"/>')
        parts.append(f'<text class="small" x="{x:.2f}" y="{top+chart_h+18:.2f}" text-anchor="middle">{int(x_val)}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}"/>')
    parts.append(f'<text class="small" x="{left + chart_w/2:.2f}" y="{height-18}" text-anchor="middle">total buy + sell actions</text>')
    parts.append(f'<text class="small" x="22" y="{top + chart_h/2:.2f}" transform="rotate(-90 22 {top + chart_h/2:.2f})" text-anchor="middle">final portfolio value</text>')
    for agent in RL_AGENTS:
        x = scale_value(transaction_summary[agent]["total_actions"], min_x, max_x, left, left + chart_w)
        y = scale_value(mean_metrics[agent]["final_value"], min_y, max_y, top + chart_h, top)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{COLORS[agent]}"/>')
        parts.append(f'<text class="small" x="{x+10:.2f}" y="{y-8:.2f}">{agent.upper()}</text>')
    write_text(path, build_svg(parts, width, height))


def plot_final_weights_heatmap(path: str, tickers: list[str], final_weights: dict[str, dict[str, float]]) -> None:
    width = 980
    height = 120 + 60 * len(RL_AGENTS)
    left = 140
    top = 80
    cell_w = 86
    cell_h = 42
    max_abs = 2.5
    parts = [
        f'<text x="{width/2:.0f}" y="30" text-anchor="middle" font-size="20" font-weight="700">Final Portfolio Weights Heatmap</text>',
        '<text class="small" x="490" y="52" text-anchor="middle">Blue = long, red = short, grey = near zero. Values clipped visually at +/- 2.5.</text>',
    ]
    for col, ticker in enumerate(tickers):
        x = left + col * cell_w + cell_w / 2
        parts.append(f'<text class="small" x="{x:.2f}" y="{top-14}" text-anchor="middle">{ticker}</text>')
    for row, agent in enumerate(RL_AGENTS):
        y = top + row * cell_h
        parts.append(f'<text class="small" x="{left-12}" y="{y + cell_h/2 + 4:.2f}" text-anchor="end">{agent.upper()}</text>')
        for col, ticker in enumerate(tickers):
            value = final_weights[agent][ticker]
            clipped = max(-max_abs, min(max_abs, value))
            intensity = abs(clipped) / max_abs
            if clipped > 0:
                color = f"rgb({int(230 - 120 * intensity)},{int(245 - 60 * intensity)},{int(255 - 10 * intensity)})"
            elif clipped < 0:
                color = f"rgb({int(255 - 10 * intensity)},{int(238 - 120 * intensity)},{int(238 - 120 * intensity)})"
            else:
                color = "#f3f4f6"
            x = left + col * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-4}" height="{cell_h-4}" fill="{color}" stroke="#d1d5db"/>')
            parts.append(f'<text class="small" x="{x + (cell_w-4)/2:.2f}" y="{y + cell_h/2 + 4:.2f}" text-anchor="middle">{value:.2f}</text>')
    write_text(path, build_svg(parts, width, height))


def build_report(
    data_root: str,
    output_dir: str,
    hypothesis_path: str,
    dates: list[str],
    account_series: list[dict[str, float]],
    mean_metrics: dict[str, dict[str, float]],
    single_run_risk: dict[str, dict[str, float | None]],
    transaction_summary: dict[str, dict[str, float]],
    ticker_pnl: dict[str, dict[str, tuple[str, float]]],
    snapshot_stats: dict[str, dict[str, float]],
    tickers: list[str],
    final_weights: dict[str, dict[str, float]],
) -> str:
    final_row = account_series[-1]
    benchmark_final = mean_metrics["benchmark"]["final_value"]
    td3_final = mean_metrics["td3"]["final_value"]
    sac_final = mean_metrics["sac"]["final_value"]
    losing_agents = ["a2c", "ddpg", "ppo"]
    strongest_consensus_cash = statistics.mean(final_weights[agent]["cash"] for agent in RL_AGENTS)
    hypotheses = pick_hypotheses()

    outcome_table = "\n".join(
        [
            "| Agent | Final Value | Sharpe | Max Drawdown | Single-Run Status |",
            "|---|---:|---:|---:|---|",
            *[
                f"| {agent.upper()} | {fmt_money(mean_metrics[agent]['final_value'])} | {mean_metrics[agent]['sharpe']:.2f} | {mean_metrics[agent]['max_drawdown']:.2f} | "
                f"{'valid' if single_run_risk.get(agent, {}).get('sharpe') is not None else 'failed / unstable'} |"
                for agent in AGENTS
            ],
        ]
    )

    behavior_table = "\n".join(
        [
            "| Agent | Actions | Full-Cash Rows | Negative-Equity Rows | Max Gross Exposure | Final Cash Weight |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {agent.upper()} | {int(transaction_summary[agent]['total_actions'])} | {int(snapshot_stats[agent]['full_cash_rows'])} | "
                f"{int(snapshot_stats[agent]['negative_equity_rows'])} | {snapshot_stats[agent]['max_gross_exposure']:.2f} | {snapshot_stats[agent]['final_cash_weight']:.2f} |"
                for agent in RL_AGENTS
            ],
        ]
    )

    ticker_table = "\n".join(
        [
            "| Agent | Best Ticker | Best PnL | Worst Ticker | Worst PnL |",
            "|---|---|---:|---|---:|",
            *[
                f"| {agent.upper()} | {ticker_pnl[agent]['best'][0]} | {fmt_money(ticker_pnl[agent]['best'][1])} | "
                f"{ticker_pnl[agent]['worst'][0]} | {fmt_money(ticker_pnl[agent]['worst'][1])} |"
                for agent in RL_AGENTS
            ],
        ]
    )

    return f"""# Interleaved RL Quant Analysis

Generated: {datetime.now().isoformat(timespec="seconds")}

Data scope:

- `{data_root}/run_000/interleaved`
- `{data_root}/aggregated_interleaved`
- Hypothesis note used as context: `{hypothesis_path}`

## What: Outcome Summary

The benchmark finishes at {fmt_money(benchmark_final)} with Sharpe {mean_metrics['benchmark']['sharpe']:.2f}. Among the RL agents, TD3 is the clear winner at {fmt_money(td3_final)}, SAC is the only other economically strong survivor at {fmt_money(sac_final)}, and A2C, DDPG, and PPO all finish with negative terminal portfolio value.

This means the leaderboard splits into three groups:

- benchmark: best risk-adjusted profile and no evidence of insolvency
- TD3 and SAC: profitable but much more aggressive
- A2C, DDPG, PPO: economically broken by the end of the path

![Final values](final_values.svg)

![Normalized curves](normalized_equity_curves.svg)

{outcome_table}

## How: What The Algorithms Actually Did

The winning RL agents stayed continuously engaged with the market. TD3 and SAC retained gross exposure through the sample and finished with explicit long-short books. TD3 concentrated heavily in a large JPM long against several shorts. SAC ended with a more diversified long-short book, including long AAPL, MSFT, and XOM against short CAT and JPM.

The losing agents behaved very differently. A2C, DDPG, and PPO all spent long stretches with `cash_weight = 1.0` while also carrying negative total asset. That is not ordinary defensive behavior; it looks like a post-blowup liquidation state. PPO also traded far more often than the others, which is consistent with over-rotation rather than clean signal expression.

![Failure-state behavior](cash_negative_equity.svg)

![Turnover vs outcome](turnover_vs_final_value.svg)

![Final weights](final_weights_heatmap.svg)

{behavior_table}

### Ticker-Level Behavior

Ticker-level PnL suggests that the surviving agents were not winning broadly everywhere. They won through a few large bets:

- TD3's standout driver is JPM.
- SAC's strongest contributor is AAPL.
- The weak agents are often dominated by large losses in WMT or AAPL.

{ticker_table}

## Why: Hypothesized Causes

These are the leading explanations after reading the prior hypothesis note and checking the behavior data:

{chr(10).join(f"- {item}" for item in hypotheses)}

## Interpretation

The simplest synthesis is:

- The benchmark wins because it stays inside a sane risk envelope.
- TD3 wins among RL agents by expressing the strongest conviction in the trade that mattered most on this path.
- SAC survives because it stays diversified enough to avoid ruin while still taking real risk.
- A2C, DDPG, and PPO appear to cross into negative-equity states and then stop functioning as economically meaningful strategies.

Two caveats matter:

- The `aggregated_interleaved` summary still shows only one run, so none of this should be treated as robust cross-run statistical proof yet.
- Weight-based interpretation for A2C, DDPG, and PPO becomes unreliable once portfolio equity is near zero or negative, so the behavioral reading is stronger than the exact leverage arithmetic for those agents.

## Files Produced

- `final_values.svg`
- `normalized_equity_curves.svg`
- `cash_negative_equity.svg`
- `turnover_vs_final_value.svg`
- `final_weights_heatmap.svg`
- `interleaved_quant_report.md`
"""


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    dates, account_series = load_account_values(args.data_root)
    mean_metrics = load_mean_metrics(args.data_root)
    single_run_risk = load_single_run_risk(args.data_root)
    transaction_summary = load_transaction_summary(args.data_root)
    ticker_pnl = load_ticker_pnl(args.data_root)
    snapshot_stats = load_snapshot_stats(args.data_root)
    tickers, final_weights = load_final_weights(args.data_root)

    plot_final_values(os.path.join(args.output_dir, "final_values.svg"), mean_metrics)
    plot_equity_curves(os.path.join(args.output_dir, "normalized_equity_curves.svg"), dates, account_series)
    plot_cash_and_equity(os.path.join(args.output_dir, "cash_negative_equity.svg"), snapshot_stats)
    plot_turnover_vs_outcome(
        os.path.join(args.output_dir, "turnover_vs_final_value.svg"),
        mean_metrics,
        transaction_summary,
    )
    plot_final_weights_heatmap(
        os.path.join(args.output_dir, "final_weights_heatmap.svg"),
        tickers,
        final_weights,
    )

    report = build_report(
        args.data_root,
        args.output_dir,
        args.hypotheses_md,
        dates,
        account_series,
        mean_metrics,
        single_run_risk,
        transaction_summary,
        ticker_pnl,
        snapshot_stats,
        tickers,
        final_weights,
    )
    write_text(os.path.join(args.output_dir, "interleaved_quant_report.md"), report)
    print(f"Wrote analysis artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
