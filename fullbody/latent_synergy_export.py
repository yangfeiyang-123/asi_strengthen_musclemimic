"""Export checkpoint-bound offline evidence for latent/synergy analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from musclemimic.latent_muscle.analysis_export import export_analysis_inputs
from musclemimic.latent_muscle.phase_contract import load_phase_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--val-dataset-dir", type=Path, required=True)
    parser.add_argument("--synergy-basis", type=Path, required=True)
    parser.add_argument("--synergy-basis-fingerprint", required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--max-intervention-directions", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=[-1.0, -0.5, 0.5, 1.0],
    )
    parser.add_argument("--require-all-phases", action="store_true", default=False)
    parser.add_argument("--phase-contract-json", type=Path, default=None)
    parser.add_argument("--causal-interventions-npz", type=Path, default=None)
    parser.add_argument("--causal-interventions-manifest", type=Path, default=None)
    parser.add_argument(
        "--require-causal-interventions",
        action="store_true",
        default=False,
        help="Fail unless a verified, separately sealed environment-rollout artifact is supplied.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = export_analysis_inputs(
        latent_checkpoint=args.latent_checkpoint,
        dataset_dir=args.dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        synergy_basis=args.synergy_basis,
        synergy_basis_fingerprint=str(args.synergy_basis_fingerprint),
        output_npz=args.output_npz,
        max_samples=int(args.max_samples),
        max_intervention_directions=int(args.max_intervention_directions),
        epsilons=tuple(float(value) for value in args.epsilon),
        batch_size=int(args.batch_size),
        require_all_phases=bool(args.require_all_phases),
        phase_contract=(
            None
            if args.phase_contract_json is None
            else load_phase_contract(args.phase_contract_json)
        ),
        causal_interventions_npz=args.causal_interventions_npz,
        causal_interventions_manifest=args.causal_interventions_manifest,
        require_causal_interventions=bool(args.require_causal_interventions),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
