"""Display and headless rendering utilities."""

import os
import subprocess
import sys
from glob import glob


def detect_headless_environment() -> bool:
    """
    Detect if running in a headless environment (no display available).

    Returns:
        True if headless rendering should be used, False otherwise.
    """

    if sys.platform in ("darwin", "win32"):
        return False

    if "DISPLAY" not in os.environ:
        return True

    try:
        result = subprocess.run(["xset", "q"], capture_output=True, text=True, timeout=2)
        return result.returncode != 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return True


def setup_headless_rendering() -> None:
    """
    Set up environment for headless rendering.

    This configures MuJoCo for offscreen rendering when no display is
    available. EGL is preferred when render devices are accessible; OSMesa is
    used as a software fallback when DRI devices exist but are not readable by
    the current user.
    """
    backend = os.environ.get("MUJOCO_GL")
    if backend is None:
        backend = "osmesa" if _dri_devices_exist_without_access() else "egl"
        os.environ["MUJOCO_GL"] = backend

    if backend == "egl":
        print("No display detected - enabling headless rendering with EGL")
        print("   Set MUJOCO_GL=egl for headless rendering")
    elif backend == "osmesa":
        print("No display detected - enabling headless rendering with OSMesa")
        print("   Set MUJOCO_GL=osmesa for software rendering")
    else:
        print(f"No display detected - preserving MUJOCO_GL={backend} for headless rendering")


def setup_headless_rendering_if_needed() -> None:
    """
    Automatically set up headless rendering if no display is available.

    This is a convenience function that combines detection and setup.
    """
    if detect_headless_environment():
        setup_headless_rendering()


def _dri_devices_exist_without_access() -> bool:
    """Return True when DRI render devices exist but none are accessible."""
    paths = sorted(glob("/dev/dri/renderD*") + glob("/dev/dri/card*"))
    if not paths:
        return False
    return not any(os.access(path, os.R_OK | os.W_OK) for path in paths)
