#!/usr/bin/env python3
"""Generate the data-driven paper figures from artifacts/metrics/*.

Fig. 1 (method diagram) is not generated here -- it is a schematic, not a plot
of data, and belongs in the paper draft as a hand-made or slide-tool diagram.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rap2p.config import load_and_prepare  # noqa: E402


def figure_adaptation_curve(metrics_dir: Path, figures_dir: Path) -> None:
    """Fig. 2: accuracy / MAE / NLL vs K, macro domain, split=test/seen."""
    macro = pd.read_parquet(metrics_dir / "respondent_macro.parquet")
    macro = macro[macro["domain"].eq("macro") & macro["split"].eq("test") & macro["item_pool"].eq("seen")]
    methods = sorted(macro["method"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for metric, ax, title in zip(["accuracy", "mae", "nll"], axes, ["Accuracy ↑", "Ordinal MAE ↓", "NLL ↓"]):
        for method in methods:
            series = macro[macro["method"].eq(method)].sort_values("k")
            if series.empty:
                continue
            ax.plot(series["k"], series[metric], marker="o", label=method, linewidth=1.5)
        ax.set_xlabel("K (known answers)")
        ax.set_title(title)
    axes[0].legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_adaptation_curve.pdf")
    fig.savefig(figures_dir / "fig2_adaptation_curve.png", dpi=200)
    plt.close(fig)


def figure_correlation_heatmaps(predictions_dir: Path, figures_dir: Path, methods: list[str], k: int = 5) -> None:
    """Fig. 3: human vs. predicted item-item correlation matrices, expected-score based."""
    from rap2p.eval.metrics import _expected_scores, load_predictions

    predictions = load_predictions(predictions_dir)
    predictions = predictions[predictions["split"].eq("test") & predictions["item_pool"].eq("seen") & predictions["k"].eq(k)]
    predictions = predictions.copy()
    predictions["expected_score"] = _expected_scores(predictions)
    predictions["human_score"] = predictions["answer_index"] / predictions["n_options"].sub(1).clip(lower=1)

    domain = sorted(predictions["domain"].unique())[0]
    domain_predictions = predictions[predictions["domain"].eq(domain)]
    human_wide = domain_predictions.drop_duplicates(["panel_id", "question_key"]).pivot_table(
        index="panel_id", columns="question_key", values="human_score"
    )
    human_corr = human_wide.corr()

    panels_to_plot = ["Human", *methods]
    fig, axes = plt.subplots(1, len(panels_to_plot), figsize=(4 * len(panels_to_plot), 4))
    axes = axes if hasattr(axes, "__len__") else [axes]
    axes[0].imshow(human_corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title("Human")
    for ax, method in zip(axes[1:], methods):
        method_frame = domain_predictions[domain_predictions["method"].eq(method)]
        wide = method_frame.pivot_table(index="panel_id", columns="question_key", values="expected_score")
        wide = wide.reindex(columns=human_corr.columns)
        corr = wide.corr()
        ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(method)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Item-item correlation, K={k}, domain={domain}")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_correlation_heatmaps.pdf")
    fig.savefig(figures_dir / "fig3_correlation_heatmaps.png", dpi=200)
    plt.close(fig)


def figure_ablation_forest(metrics_dir: Path, figures_dir: Path) -> None:
    """Fig. 6 (ablation forest plot): bootstrap accuracy deltas relative to Context QLoRA."""
    bootstrap_path = metrics_dir / "bootstrap_accuracy.csv"
    if not bootstrap_path.exists():
        print(f"[skip] {bootstrap_path} not found, run evaluate_all.py first")
        return
    bootstrap = pd.read_csv(bootstrap_path)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(bootstrap) + 1))
    y_positions = range(len(bootstrap))
    ax.errorbar(
        bootstrap["difference"], list(y_positions),
        xerr=[bootstrap["difference"] - bootstrap["ci_low"], bootstrap["ci_high"] - bootstrap["difference"]],
        fmt="o", capsize=3,
    )
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([f"{row.method} vs {row.reference} (K={row.k})" for row in bootstrap.itertuples()], fontsize=7)
    ax.set_xlabel("Accuracy difference (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig6_bootstrap_forest.pdf")
    fig.savefig(figures_dir / "fig6_bootstrap_forest.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_and_prepare(args.config)
    metrics_dir = Path(config["paths"]["metrics"])
    predictions_dir = Path(config["paths"]["predictions"])
    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_adaptation_curve(metrics_dir, figures_dir)
    figure_correlation_heatmaps(predictions_dir, figures_dir, methods=["context_qlora_seed1701", "p2p_static_seed1701", "rap2p_seed1701"])
    figure_ablation_forest(metrics_dir, figures_dir)
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
