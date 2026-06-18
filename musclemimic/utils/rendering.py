"""Rendering helpers shared by MuJoCo visualization scripts."""

from __future__ import annotations

import mujoco
import numpy as np


def disable_skybox_textures(model: mujoco.MjModel, rgb: tuple[int, int, int] = (255, 255, 255)) -> int:
    """Replace skybox texture pixels with a flat color.

    MuJoCo keeps compiled texture data in a flat ``tex_rgb`` buffer. Replacing
    only textures of type ``mjTEXTURE_SKYBOX`` removes image backgrounds while
    leaving body/mesh textures intact.
    """
    if getattr(model, "ntex", 0) == 0 or not hasattr(model, "tex_rgb"):
        return 0

    color = np.asarray(rgb, dtype=model.tex_rgb.dtype)
    disabled = 0
    tex_rgb_len = len(model.tex_rgb)
    for tex_id in range(model.ntex):
        if int(model.tex_type[tex_id]) != int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            continue

        start = int(model.tex_adr[tex_id])
        end = int(model.tex_adr[tex_id + 1]) if tex_id + 1 < model.ntex else tex_rgb_len
        if start < 0 or end <= start:
            continue

        skybox_pixels = model.tex_rgb[start:end].reshape(-1, 3)
        skybox_pixels[:] = color
        disabled += 1

    return disabled
