from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_json


def find_candidate_entrypoints(moshpp_dir: str | Path) -> list[str]:
    root = Path(moshpp_dir)
    if not root.exists():
        return []
    names: list[str] = []
    interesting_ext = {".py", ".sh", ".md", ".rst", ".txt", ".yaml", ".yml"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in interesting_ext:
            continue
        rel = str(path.relative_to(root))
        low = rel.lower()
        if any(k in low for k in ("readme", "tutorial", "example", "demo", "mosh", "fit", "run")):
            names.append(rel)
    return sorted(names)[:200]


def infer_command(
    c3d: str | Path,
    config_dir: str | Path,
    moshpp_dir: str | Path,
    conda_env: str | None,
    out_dir: str | Path,
) -> list[str] | None:
    root = Path(moshpp_dir)
    possible = [
        root / "scripts" / "moshpp.py",
        root / "scripts" / "run_moshpp.py",
        root / "src" / "moshpp" / "mosh_head.py",
    ]
    entry = next((p for p in possible if p.exists()), None)
    if entry is None:
        return None
    base = ["python", str(entry), "--config", str(Path(config_dir) / "fit_config.yaml")]
    if conda_env:
        return ["conda", "run", "-n", conda_env, *base]
    return base


def run_moshpp(
    c3d: str | Path,
    config_dir: str | Path,
    moshpp_dir: str | Path,
    conda_env: str | None,
    out_dir: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    out = ensure_dir(out_dir)
    candidates = find_candidate_entrypoints(moshpp_dir)
    (out / "candidate_entrypoints.txt").write_text("\n".join(candidates), encoding="utf-8")

    command = infer_command(c3d, config_dir, moshpp_dir, conda_env, out_dir)
    if command is None:
        message = (
            "Could not determine an official MoSh++ run entrypoint. "
            "Review candidate_entrypoints.txt and run the official command manually or adapt run_moshpp.py."
        )
        (out / "moshpp_stdout.log").write_text("", encoding="utf-8")
        (out / "moshpp_stderr.log").write_text(message + "\n", encoding="utf-8")
        write_json(out / "run_report.json", {"ok": False, "error": message, "candidates": candidates})
        raise RuntimeError(message)

    shell_line = " ".join(f'"{x}"' if " " in str(x) else str(x) for x in command)
    (out / "run_command.sh").write_text(shell_line + "\n", encoding="utf-8")
    if dry_run:
        report = {"ok": True, "dry_run": True, "command": command, "candidates": candidates}
        write_json(out / "run_report.json", report)
        return report

    env = os.environ.copy()
    env["MOSHPP_C3D"] = str(c3d)
    env["MOSHPP_CONFIG_DIR"] = str(config_dir)
    env["MOSHPP_OUT_DIR"] = str(out)
    proc = subprocess.run(command, cwd=str(moshpp_dir), env=env, capture_output=True, text=True)
    (out / "moshpp_stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (out / "moshpp_stderr.log").write_text(proc.stderr, encoding="utf-8", errors="replace")
    report = {"ok": proc.returncode == 0, "returncode": proc.returncode, "command": command, "candidates": candidates}
    write_json(out / "run_report.json", report)
    if proc.returncode != 0:
        raise RuntimeError(f"MoSh++ command failed with return code {proc.returncode}. See logs in {out}.")
    return report
