"""Plot one fitted synergy basis: muscle composition beside activation timing.

The per-fit plots written by ``extract_synergy`` answer "how well does K
reconstruct"; this one answers "what are the synergies".  Weights and
activations are drawn together because neither reads on its own -- a weight
column only becomes interpretable once you know when in the movement it is
recruited.
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
from emg.storage import read_json


def plot_basis(fit_dir: Path, output_path: Path, *, title: str | None = None) -> Path:
    with np.load(fit_dir / "synergy_basis.npz", allow_pickle=False) as payload:
        basis = np.asarray(payload["W"], dtype=np.float64)
        activation = np.asarray(payload["H"], dtype=np.float64)
        sensor_ids = [int(value) for value in payload["channel_ids"]]
        sides = [str(value) for value in payload["sides"]]
        slugs = [str(value) for value in payload["muscle_slugs"]]
    metadata = read_json(fit_dir / "synergy_metadata.json")
    rank = basis.shape[1]
    samples = int(metadata.get("time_normalize_samples") or 0)

    labels = [
        f"S{sensor:02d} {side[0].upper()}-{slug.replace('_', ' ')}"
        for sensor, side, slug in zip(sensor_ids, sides, slugs)
    ]
    basis = basis / np.maximum(np.linalg.norm(basis, axis=0, keepdims=True), np.finfo(float).eps)

    # Average the activation over trials so the profile shows the recruitment
    # shared by the repetitions rather than one trial's timing.
    trials = activation.shape[1] // samples if samples else 1
    if trials >= 1 and samples and trials * samples == activation.shape[1]:
        profile = activation.reshape(rank, trials, samples)
        mean_profile = profile.mean(axis=1)
        spread = profile.std(axis=1)
        phase = np.linspace(0.0, 100.0, samples)
    else:
        mean_profile = activation
        spread = np.zeros_like(activation)
        phase = np.linspace(0.0, 100.0, activation.shape[1])

    order = np.argsort(mean_profile.argmax(axis=1))
    basis = basis[:, order]
    mean_profile = mean_profile[order]
    spread = spread[order]

    figure, axes = plt.subplots(
        rank, 2, figsize=(11, 2.3 * rank + 1.2), gridspec_kw={"width_ratios": [1.35, 1.0]}, squeeze=False
    )
    positions = np.arange(len(labels))
    for index in range(rank):
        weights = basis[:, index]
        bar_axis = axes[index][0]
        colours = ["tab:red" if value >= 0.30 else "tab:blue" for value in weights]
        bar_axis.barh(positions, weights, color=colours)
        bar_axis.set_yticks(positions, labels, fontsize=7)
        bar_axis.invert_yaxis()
        bar_axis.set_xlim(0, max(0.35, float(weights.max()) * 1.1))
        bar_axis.axvline(0.30, color="0.5", linestyle=":", linewidth=1)
        bar_axis.set_ylabel(f"Synergy {index + 1}", fontsize=10, fontweight="bold")
        bar_axis.grid(axis="x", alpha=0.25)
        if index == 0:
            bar_axis.set_title("Muscle weights (L2-normalised column)", fontsize=10)

        line_axis = axes[index][1]
        line_axis.plot(phase, mean_profile[index], color="tab:green")
        line_axis.fill_between(
            phase,
            np.maximum(mean_profile[index] - spread[index], 0.0),
            mean_profile[index] + spread[index],
            color="tab:green",
            alpha=0.2,
        )
        peak = phase[int(mean_profile[index].argmax())]
        line_axis.axvline(peak, color="tab:green", linestyle="--", linewidth=1)
        line_axis.set_xlim(0, 100)
        line_axis.grid(alpha=0.25)
        line_axis.set_ylabel("activation", fontsize=8)
        line_axis.annotate(
            f"peak {peak:.0f}%",
            xy=(peak, mean_profile[index].max()),
            xytext=(4, -8),
            textcoords="offset points",
            fontsize=8,
        )
        if index == 0:
            line_axis.set_title("Activation across the cued window (mean ± SD over trials)", fontsize=10)
        if index == rank - 1:
            line_axis.set_xlabel("% of cued window")
            bar_axis.set_xlabel("weight")

    heading = title or f"{'/'.join(metadata.get('actions', []) or ['session'])} — K={rank}"
    figure.suptitle(
        f"{heading}\n"
        f"global VAF {metadata['global_vaf']:.3f} (balanced) / {metadata['recorded_unit_global_vaf']:.3f} "
        f"(recorded units), channel normalisation "
        f"{metadata['nmf_parameters']['channel_normalization']}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    print(f"Figure saved: {plot_basis(args.fit_dir, args.output, title=args.title)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
