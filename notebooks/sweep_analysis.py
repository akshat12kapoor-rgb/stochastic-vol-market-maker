"""Integration & Analysis Agent -- multi-seed backtest sweep and regime case studies.

Run with: `source .venv/bin/activate && python -m notebooks.sweep_analysis`
(run from the project root so `backtest`/`market_maker`/`vol_models`/`pricing`
are importable as top-level packages, same convention as
`vol_models/comparison.py` and `backtest/comparison.py`).

This is a REPORT-GENERATION script (mirrors `vol_models/comparison.py` and
`backtest/comparison.py`'s convention), not a pytest test -- it produces the
final integration analysis for the whole project:

1. A 40-seed sweep of `backtest.engine.run_backtest` for both quoting models
   (`market_seed` = 1..40, `arrival_seed` = market_seed + 1000), collecting
   final P&L / Sharpe / max-drawdown distributions -- NOT just the single
   `market_seed=42` path `backtest/comparison.py` reports.
2. Two regime case studies (one seed where Heston wins clearly, one where BS
   wins clearly) with plain-English explanation tied to the realized
   variance/spot path.
3. All figures saved as PNGs under `data/`; all numbers echoed to stdout AND
   written to `data/sweep_results.csv` so the results are inspectable without
   re-running anything.

The Heston quoter's calibrated params and the BS quoter's flat vol are
computed ONCE (both are deterministic functions of `TRUE_MARKET_PARAMS`
alone, independent of `market_seed`/`arrival_seed` -- see
`backtest.quoter_calibration`) and reused across all 40 seeds, both to save
~1s/seed of redundant optimizer calls and because that's what "calibrate
once, trade the quarter" (the project's own framing) means applied across
seeds: it's the same desk, quoting on 40 different realized quarters, not a
re-calibrated desk each time.
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtest.engine import run_backtest
from backtest.market_sim import TRUE_MARKET_PARAMS
from backtest.quoter_calibration import calibrate_heston_quoter_params, compute_flat_vol_for_bs_quoter

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

N_SEEDS = 40
SEEDS = list(range(1, N_SEEDS + 1))


def run_sweep():
    flat_vol = compute_flat_vol_for_bs_quoter(TRUE_MARKET_PARAMS)
    heston_params = calibrate_heston_quoter_params(TRUE_MARKET_PARAMS)
    print(f"BS flat_vol = {flat_vol:.4f}")
    print(f"Heston quoter params = {heston_params}")

    rows = []
    results_cache = {}  # seed -> (r_bs, r_heston), kept only for the 2 case-study seeds later
    for seed in SEEDS:
        arrival_seed = seed + 1000
        r_bs = run_backtest(
            "black_scholes", market_seed=seed, arrival_seed=arrival_seed,
            flat_vol=flat_vol, heston_quoter_params=heston_params,
        )
        r_h = run_backtest(
            "heston", market_seed=seed, arrival_seed=arrival_seed,
            flat_vol=flat_vol, heston_quoter_params=heston_params,
        )
        rows.append({
            "seed": seed,
            "pnl_bs": r_bs.pnl[-1], "pnl_heston": r_h.pnl[-1],
            "sharpe_bs": r_bs.sharpe, "sharpe_heston": r_h.sharpe,
            "maxdd_bs": r_bs.max_drawdown, "maxdd_heston": r_h.max_drawdown,
            "nfills_bs": r_bs.meta["n_fills"], "nfills_heston": r_h.meta["n_fills"],
            "final_spot": r_bs.spot_path[-1], "spot0": r_bs.spot_path[0],
            "terminal_variance": r_bs.variance_path[-1],
            "mean_variance": float(np.mean(r_bs.variance_path)),
            "vol_of_var": float(np.std(np.diff(r_bs.variance_path))),
            "max_spot_drawdown": float(np.min(r_bs.spot_path / np.maximum.accumulate(r_bs.spot_path) - 1.0)),
        })
        results_cache[seed] = (r_bs, r_h)
        print(f"seed={seed:2d}  BS pnl={r_bs.pnl[-1]:8.2f} sharpe={r_bs.sharpe:6.2f}  |  "
              f"Heston pnl={r_h.pnl[-1]:8.2f} sharpe={r_h.sharpe:6.2f}  |  "
              f"winner={'BS' if r_bs.pnl[-1] > r_h.pnl[-1] else 'Heston'}")

    return rows, results_cache, flat_vol, heston_params


def summarize(rows):
    pnl_bs = np.array([r["pnl_bs"] for r in rows])
    pnl_h = np.array([r["pnl_heston"] for r in rows])
    sharpe_bs = np.array([r["sharpe_bs"] for r in rows])
    sharpe_h = np.array([r["sharpe_heston"] for r in rows])
    maxdd_bs = np.array([r["maxdd_bs"] for r in rows])
    maxdd_h = np.array([r["maxdd_heston"] for r in rows])

    win_rate_bs = float(np.mean(pnl_bs > pnl_h))

    summary = {
        "n_seeds": len(rows),
        "pnl_bs_mean": float(np.mean(pnl_bs)), "pnl_bs_median": float(np.median(pnl_bs)), "pnl_bs_std": float(np.std(pnl_bs, ddof=1)),
        "pnl_heston_mean": float(np.mean(pnl_h)), "pnl_heston_median": float(np.median(pnl_h)), "pnl_heston_std": float(np.std(pnl_h, ddof=1)),
        "sharpe_bs_mean": float(np.mean(sharpe_bs)), "sharpe_heston_mean": float(np.mean(sharpe_h)),
        "maxdd_bs_mean": float(np.mean(maxdd_bs)), "maxdd_heston_mean": float(np.mean(maxdd_h)),
        "maxdd_bs_median": float(np.median(maxdd_bs)), "maxdd_heston_median": float(np.median(maxdd_h)),
        "win_rate_bs": win_rate_bs, "win_rate_heston": 1.0 - win_rate_bs,
        "mean_pnl_gap_bs_minus_heston": float(np.mean(pnl_bs - pnl_h)),
        "paired_pnl_diff_std": float(np.std(pnl_bs - pnl_h, ddof=1)),
    }
    # paired t-stat (BS - Heston), purely descriptive -- not claiming formal significance
    diffs = pnl_bs - pnl_h
    se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    summary["paired_diff_mean"] = float(np.mean(diffs))
    summary["paired_diff_se"] = float(se)
    summary["paired_t_stat"] = float(np.mean(diffs) / se) if se > 0 else float("nan")

    print("\n=== Summary over", len(rows), "seeds ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return summary


def write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def plot_distribution(rows, summary, output_path):
    pnl_bs = np.array([r["pnl_bs"] for r in rows])
    pnl_h = np.array([r["pnl_heston"] for r in rows])
    sharpe_bs = np.array([r["sharpe_bs"] for r in rows])
    sharpe_h = np.array([r["sharpe_heston"] for r in rows])
    maxdd_bs = np.array([r["maxdd_bs"] for r in rows])
    maxdd_h = np.array([r["maxdd_heston"] for r in rows])
    seeds = [r["seed"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: final P&L per seed (paired bar / dot plot)
    ax = axes[0, 0]
    x = np.arange(len(seeds))
    width = 0.38
    ax.bar(x - width / 2, pnl_bs, width, label="BS-quoting", color="#4C72B0")
    ax.bar(x + width / 2, pnl_h, width, label="Heston-quoting", color="#DD8452")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x[::4])
    ax.set_xticklabels([str(s) for s in seeds[::4]])
    ax.set_xlabel("market_seed")
    ax.set_ylabel("final P&L ($)")
    ax.set_title(f"Final P&L per seed ({len(seeds)} seeds)")
    ax.legend()

    # Panel 2: box-style comparison of P&L distributions
    ax = axes[0, 1]
    bp = ax.boxplot([pnl_bs, pnl_h], tick_labels=["BS", "Heston"], patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("final P&L ($)")
    ax.set_title(f"P&L distribution across {len(seeds)} seeds\n"
                 f"mean: BS \\${summary['pnl_bs_mean']:.1f} vs Heston \\${summary['pnl_heston_mean']:.1f}  |  "
                 f"BS win rate: {summary['win_rate_bs']*100:.0f}%")

    # Panel 3: Sharpe distribution
    ax = axes[1, 0]
    bp = ax.boxplot([sharpe_bs, sharpe_h], tick_labels=["BS", "Heston"], patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("annualized Sharpe")
    ax.set_title(f"Sharpe distribution\nmean: BS {summary['sharpe_bs_mean']:.2f} vs Heston {summary['sharpe_heston_mean']:.2f}")

    # Panel 4: max drawdown distribution
    ax = axes[1, 1]
    bp = ax.boxplot([maxdd_bs, maxdd_h], tick_labels=["BS", "Heston"], patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("max drawdown ($, non-positive)")
    ax.set_title(f"Max drawdown distribution\nmean: BS \\${summary['maxdd_bs_mean']:.1f} vs Heston \\${summary['maxdd_heston_mean']:.1f}")

    fig.suptitle(f"BS-quoting vs Heston-quoting market maker: {len(seeds)}-seed sweep", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_case_study(seed, r_bs, r_heston, output_path, title_note):
    times = r_bs.times
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(times, r_bs.pnl, label="BS-quoting P&L", color="#4C72B0", linewidth=1.8)
    ax.plot(times, r_heston.pnl, label="Heston-quoting P&L", color="#DD8452", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("P&L ($)")
    ax.set_title(f"Case study: market_seed={seed} -- {title_note}")
    ax.legend()

    ax = axes[1]
    ax.plot(times, r_bs.variance_path, color="#55A868", linewidth=1.5, label="realized variance")
    ax.axhline(TRUE_MARKET_PARAMS.theta, color="gray", linestyle="--", linewidth=1.0, label=f"long-run theta={TRUE_MARKET_PARAMS.theta}")
    ax.set_ylabel("realized instantaneous variance")
    ax.set_xlabel("time (years)")
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(times, r_bs.spot_path, color="#8172B2", linewidth=1.0, alpha=0.7, label="spot")
    ax2.set_ylabel("spot", color="#8172B2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    rows, cache, flat_vol, heston_params = run_sweep()
    summary = summarize(rows)
    write_csv(rows, DATA_DIR / "sweep_results.csv")
    plot_distribution(rows, summary, DATA_DIR / "sweep_pnl_distribution.png")

    # Pick case-study seeds: the largest Heston win and the largest BS win
    # (by P&L gap), among the 40 swept seeds.
    gaps = [(r["seed"], r["pnl_bs"] - r["pnl_heston"]) for r in rows]
    seed_bs_wins_big = max(gaps, key=lambda t: t[1])[0]
    seed_heston_wins_big = min(gaps, key=lambda t: t[1])[0]

    print(f"\nCase study seeds: BS wins big at seed={seed_bs_wins_big}, Heston wins big at seed={seed_heston_wins_big}")

    for seed, label in [(seed_heston_wins_big, "Heston-quoting wins"), (seed_bs_wins_big, "BS-quoting wins")]:
        r_bs, r_h = cache[seed]
        row = next(r for r in rows if r["seed"] == seed)
        note = (f"{label} (BS \\${row['pnl_bs']:.1f} vs Heston \\${row['pnl_heston']:.1f}, "
                f"vol-of-var={row['vol_of_var']:.5f}, max spot drawdown={row['max_spot_drawdown']*100:.1f}%)")
        fname = f"case_study_seed{seed}_{'heston_wins' if label.startswith('Heston') else 'bs_wins'}.png"
        plot_case_study(seed, r_bs, r_h, DATA_DIR / fname, note)

    return rows, summary, seed_bs_wins_big, seed_heston_wins_big


if __name__ == "__main__":
    main()
