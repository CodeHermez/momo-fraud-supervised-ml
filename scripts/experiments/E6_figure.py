"""E6 figure -- threshold-rule divergence across the evaluation cost ratio.

Small multiples, one panel per learner: the risk of the ROC-optimal threshold
(config B) and of the risk-optimal threshold (config C) as the evaluation cost
ratio sweeps four octaves either side of the class ratio.

Two series only, so the house categorical slots 0 and 1 carry identity and the
legend sits once at figure level. Both axes are log: R spans 12 to 49,517 and
NER spans 0.007 to 6.4, and a linear axis would collapse either one. A single y
axis throughout -- both series are the same measure.

Reads ``results/E6_paired_by_level.csv``. Writes nothing else.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from momo_fraud import viz

ROOT = Path(__file__).resolve().parents[2]

LEARNER_LABELS = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}

#: Slot 0 is the rule the study uses; slot 1 the one it is compared against.
COLOR_C = viz.CATEGORICAL[0]
COLOR_B = viz.CATEGORICAL[1]


def main() -> int:
    viz.apply_house_style()
    wide = pd.read_csv(ROOT / "results" / "E6_paired_by_level.csv")

    class_ratio = float(
        wide.loc[wide["R_level"] == "val_class_ratio", "R_eval"].iloc[0])

    panel = (wide.groupby(["learner", "R_eval"])
             .agg(ner_B=("test_ner_B", "mean"), ner_C=("test_ner_C", "mean"))
             .reset_index())

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True, sharey=True)

    for ax, learner in zip(axes.flat, LEARNER_LABELS):
        d = panel[panel.learner == learner].sort_values("R_eval")

        # Reference lines first, so the data sits above them.
        ax.axvline(class_ratio, color=viz.INK["baseline"], linestyle="--",
                   linewidth=1.2, zorder=0)
        ax.axhline(1.0, color=viz.INK["baseline"], linestyle=":",
                   linewidth=1.2, zorder=0)

        ax.plot(d.R_eval, d.ner_B, color=COLOR_B, linewidth=2,
                marker="o", markersize=5, label="ROC-optimal (config B)",
                markeredgecolor=viz.INK["surface"], markeredgewidth=0.8)
        ax.plot(d.R_eval, d.ner_C, color=COLOR_C, linewidth=2,
                marker="o", markersize=5, label="Risk-optimal (config C)",
                markeredgecolor=viz.INK["surface"], markeredgewidth=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(LEARNER_LABELS[learner], fontsize=10,
                     color=viz.INK["primary"], loc="left")

    # Annotate the two reference lines once, on the first panel only.
    a0 = axes.flat[0]
    a0.annotate("class ratio\n(rules coincide)", xy=(class_ratio, 0.02),
                xytext=(class_ratio * 1.6, 0.013), fontsize=7.5,
                color=viz.INK["secondary"], ha="left")
    # Right-hand end of the panel: the only stretch of the NER = 1 line that no
    # series passes through.
    a0.annotate("flag nothing", xy=(40000, 1.0), xytext=(40000, 1.18),
                fontsize=7.5, color=viz.INK["secondary"], ha="right")

    for ax in axes[1]:
        ax.set_xlabel("Evaluation cost ratio $R$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized Expected Risk", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.055), fontsize=9)

    fig.suptitle(
        "The two threshold rules are one rule at the class ratio, and diverge "
        "either side of it",
        fontsize=11.5, color=viz.INK["primary"], x=0.5, y=0.985)
    fig.text(0.5, 0.935,
             "Mean over 104 frozen E3 cells. Thresholds selected on validation, "
             "applied once to test. Lower is better.",
             ha="center", fontsize=8.5, color=viz.INK["secondary"])

    fig.tight_layout(rect=(0, 0.075, 1, 0.925))
    paths = viz.save_figure(fig, "E6_threshold_divergence")
    for ext, p in paths.items():
        print(f"  wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
