from __future__ import annotations

import os
import sys
from pathlib import Path


def _preferred_cuda_library_dirs() -> list[str]:
    """Return CUDA library directories shipped inside the active virtualenv."""
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.exists():
        return []
    return sorted(
        str(path)
        for path in nvidia_root.glob("*/lib")
        if path.is_dir()
    )


def _merged_ld_library_path() -> str | None:
    """Build the desired LD_LIBRARY_PATH with virtualenv CUDA libs first."""
    lib_dirs = _preferred_cuda_library_dirs()
    compat_dir = os.environ.get("MM_CUDA_COMPAT_DIR")
    if compat_dir and Path(compat_dir).is_dir():
        lib_dirs = [compat_dir, *lib_dirs]
    if not lib_dirs:
        return None

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [part for part in current.split(":") if part]
    merged: list[str] = []
    for part in [*lib_dirs, *current_parts]:
        if part not in merged:
            merged.append(part)
    return ":".join(merged)


def configure_cuda_library_path() -> None:
    """Prefer CUDA libraries shipped inside the active virtualenv.

    JAX CUDA wheels bundle NVIDIA shared libraries under
    `<venv>/lib/pythonX.Y/site-packages/nvidia/*/lib`. On machines with older
    system CUDA installs exposed via LD_LIBRARY_PATH, the dynamic loader may
    pick those older libraries first and fail JAX plugin initialization.

    This function prepends the virtualenv-provided NVIDIA library directories so
    JAX resolves the wheel-matched libraries before any system CUDA paths.
    """
    merged = _merged_ld_library_path()
    if merged is not None:
        os.environ["LD_LIBRARY_PATH"] = merged


def reexec_with_configured_cuda_env() -> None:
    """Re-exec the current Python process if CUDA library search order is wrong.

    `LD_LIBRARY_PATH` needs to be correct before the process starts, otherwise
    dlopen may still resolve system CUDA libraries first. If we detect that the
    active process was started with a different value, re-exec once with the
    corrected environment.
    """

    desired = _merged_ld_library_path()

    # JAX preallocates most GPU memory by default, which starves Warp allocations.
    # Keep this opt-out unless the user explicitly overrides it.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if desired is None:
        return

    if os.environ.get("LD_LIBRARY_PATH") == desired:
        return

    if os.environ.get("MUSCLEMIMIC_CUDA_ENV_REEXEC") == "1":
        os.environ["LD_LIBRARY_PATH"] = desired
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = desired
    env["MUSCLEMIMIC_CUDA_ENV_REEXEC"] = "1"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
