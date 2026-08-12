"""Plot the cross-action synergy reuse matrix from a study summary.

Reads ``synergy_study_summary.json`` and draws, for every ordered pair of
actions, how much of the target action a basis fitted on the source action
explains without refitting.  The diagonal is a same-data fit and is drawn only
as the ceiling each row should be read against, not as a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from emg.protocols import get_protocol
from emg.storage import read_json


def _categories(protocol_id: str, actions: list[str]) -> dict[str, str]:
    protocol = get_protocol(protocol_id)
    lookup = {action.action_id: action.category for action in protocol.actions}
    return {action: lookup.get(action, "unknown") for action in actions}


def plot_reuse(summary_path: Path, output_path: Path) -> Path:
    summary = read_json(summary_path)
    reuse = summary.get("synergy_reuse", {})
    if not reuse.get("available"):
        raise ValueError(f"{summary_path} has no usable synergy_reuse section")
    matrix_block = reuse["pairwise_reuse_matrix"]
    actions = list(matrix_block["actions"])
    category = _categories(summary["protocol_id"], actions)
    # Basic movements first so the block structure is visible at a glance.
    order = sorted(range(len(actions)), key=lambda i: (category[actions[i]] != "primitive", actions[i]))
    actions = [actions[i] for i in order]
    vaf = np.asarray(matrix_block["heldout_global_vaf"], dtype=np.float64)[np.ix_(order, order)]
    cosine = np.asarray(matrix_block["matched_basis_cosine"], dtype=np.float64)[np.ix_(order, order)]
    labels = [f"{'basic' if category[a] == 'primitive' else 'COMPLETE'}\n{a.replace('_', ' ')}" for a in actions]
    ceilings = reuse.get("within_action_heldout_ceiling", {})

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    for axis, data, title in (
        (axes[0], vaf, "Held-out global VAF\n(row basis applied to column data, no refit)"),
        (axes[1], cosine, "Matched basis cosine\n(Hungarian assignment on columns)"),
    ):
        image = axis.imshow(data, cmap="viridis", vmin=0.3, vmax=1.0)
        axis.set_xticks(range(len(actions)), labels, rotation=45, ha="right", fontsize=7)
        axis.set_yticks(range(len(actions)), labels, fontsize=7)
        axis.set_title(title, fontsize=10)
        for row in range(len(actions)):
            for column in range(len(actions)):
                value = data[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value < 0.72 else "black",
                    fontweight="bold" if row == column else "normal",
                )
        # Mark the diagonal: same data on both sides, so it bounds its row.
        for index in range(len(actions)):
            axis.add_patch(plt.Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.6))
        figure.colorbar(image, ax=axis, fraction=0.046)
    axes[0].set_ylabel("source basis")
    axes[0].set_xlabel("target data")

    note = (
        f"K={matrix_block['rank']}, channel normalisation {matrix_block['channel_normalization']}, "
        "one shared channel scale across actions. Red = same-data fit (ceiling)."
    )
    pooled = reuse.get("basic_to_complete", {})
    if pooled:
        lines = []
        for target, block in pooled.get("targets", {}).items():
            ceiling = ceilings.get(target, {})
            own = ceiling.get("mean_heldout_global_vaf")
            fraction = "" if not own else f" = {block['heldout_global_vaf'] / own:.0%} of its own held-out ceiling"
            lines.append(
                f"basic pool ({pooled['source']['trial_count']} trials) -> {target}: "
                f"VAF {block['heldout_global_vaf']:.3f}{fraction}, "
                f"{block['novelty']['novel_synergy_count']}/{block['novelty']['candidate_rank']} synergies novel"
            )
        note = note + "\n" + "\n".join(lines)
    figure.suptitle("Cross-action synergy reuse", fontsize=13)
    figure.text(0.5, 0.005, note, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"Figure saved: {plot_reuse(args.summary, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
