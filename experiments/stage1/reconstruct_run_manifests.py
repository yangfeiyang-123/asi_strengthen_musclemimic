"""Reconstruct ``manifest.json`` for transferred Stage-1 endpoint directories.

The 12 completed 800M endpoints under ``checkpoints/stage1/seed*/T*/`` were
transferred as bare Orbax leaves (``checkpoint_39063/{config,metadata,train_state}``)
without the run-level ``manifest.json`` that ``musclemimic.runner.checkpointing``
reads before any ``resume_from`` restore (muscle-control, body-synergy and
action-ABI contracts must match between run manifest, leaf config and runtime).

Every contract the validators compare is also stored verbatim inside the leaf's
``config/metadata`` (``experiment.muscle_control_contract`` etc.), so the
manifest can be rebuilt from the leaf plus the completion record
``checkpoints/stage1/COMPLETED_20260918.json`` (run_id, config_hash, source
git sha).  The rebuilt file carries a ``reconstructed`` block so it is never
mistaken for the original trainer-written manifest.

Usage (from the repo root)::

    python experiments/stage1/reconstruct_run_manifests.py           # write missing manifests
    python experiments/stage1/reconstruct_run_manifests.py --force   # overwrite reconstructed ones
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE1 = REPO / "checkpoints" / "stage1"
COMPLETED = STAGE1 / "COMPLETED_20260918.json"
CONTRACT_KEYS = ("action_manifest", "body_synergy_contract", "muscle_control_contract")


def _leaf_config(leaf: Path) -> dict:
    payload = json.loads((leaf / "config" / "metadata").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("experiment"), dict):
        raise ValueError(f"{leaf}: leaf config has no experiment mapping")
    return payload


def reconstruct(run_dir: Path, leaf: Path, record: dict | None, *, force: bool) -> str:
    target = run_dir / "manifest.json"
    if target.exists() and not force:
        existing = json.loads(target.read_text(encoding="utf-8"))
        return "kept-original" if "reconstructed" not in existing else "kept-reconstructed"
    if target.exists() and "reconstructed" not in json.loads(target.read_text(encoding="utf-8")):
        return "kept-original"
    config = _leaf_config(leaf)
    experiment = config["experiment"]
    missing = [k for k in CONTRACT_KEYS if not isinstance(experiment.get(k), dict)]
    if missing:
        raise ValueError(f"{leaf}: leaf experiment config lacks {missing}")
    manifest = {
        "config_hash": (record or {}).get("config_hash"),
        "git_sha": ((record or {}).get("source_git_sha") or "")[:12] or None,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "experiment_config": experiment,
        **{k: experiment[k] for k in CONTRACT_KEYS},
        "reconstructed": {
            "schema_version": "transferred_endpoint_manifest_reconstruction_v1",
            "reason": "endpoint transferred as a bare Orbax leaf without the trainer-written run manifest",
            "source_leaf": str(leaf.relative_to(REPO)),
            "source_completion_record": str(COMPLETED.relative_to(REPO)),
            "source_run_id": (record or {}).get("run_id"),
            "source_git_sha": (record or {}).get("source_git_sha"),
            "note": "contracts copied verbatim from the leaf config; not a trainer-written manifest",
        },
    }
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return "written"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite previously reconstructed manifests")
    args = parser.parse_args()

    records: dict[str, dict] = {}
    if COMPLETED.exists():
        for row in json.loads(COMPLETED.read_text(encoding="utf-8")).get("runs", []):
            records[str(row.get("checkpoint", "")).rsplit("/", 1)[0]] = row

    for run_dir in sorted(p for p in STAGE1.glob("seed*/T*") if p.is_dir()):
        leaves = sorted(run_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[-1]))
        if not leaves:
            continue
        key = str(run_dir.relative_to(STAGE1))
        status = reconstruct(run_dir, leaves[-1], records.get(key), force=args.force)
        print(f"{key:12s} {leaves[-1].name:18s} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
