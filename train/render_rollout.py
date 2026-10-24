"""Render a trained policy to video, with a live telemetry overlay.

The overlay is not decoration: a reach video without the distance readout
cannot be checked by a viewer, because "close enough" is exactly the
judgement being claimed. Showing the number and the 2.5 cm threshold makes
the claim falsifiable frame by frame.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import imageio.v2 as imageio
import mujoco
from PIL import Image, ImageDraw, ImageFont


def _font(size, bold=False):
    """DejaVu if present, PIL's bitmap default otherwise. The default font is
    ~11 px and unreadable once the video is scaled down, so this matters."""
    for path in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
                 f"/usr/share/fonts/truetype/liberation/LiberationSans{'-Bold' if bold else '-Regular'}.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = _font(21, bold=True)
F_BODY = _font(19)
F_TAG = _font(17, bold=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from servoarm import ServoArmReachEnv, RewardCfg, DomainRandCfg   # noqa: E402
from train.evaluate import load_policy                            # noqa: E402


def make_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.02, 0.0, 0.17]
    cam.distance = 0.60
    cam.azimuth = 138.0
    cam.elevation = -16.0
    return cam


def annotate(frame, ep, n_ep, dist_cm, inside, settled, thresh_cm):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W, _ = img.size
    d.rectangle([0, 0, W, 86], fill=(12, 14, 18, 210))
    d.text((18, 10), f"episode {ep+1}/{n_ep}", font=F_TITLE, fill=(212, 218, 228, 255))

    col = (70, 214, 128, 255) if settled else ((238, 194, 66, 255) if inside else (226, 231, 240, 255))
    d.text((18, 38), f"tip to target  {dist_cm:5.1f} cm", font=F_BODY, fill=col)
    d.text((18, 62), f"success: within {thresh_cm:.1f} cm, held 5 steps",
           font=F_TAG, fill=(150, 158, 172, 255))

    x0, y0, x1, y1 = W - 258, 34, W - 22, 52
    d.rectangle([x0, y0, x1, y1], outline=(94, 100, 114, 255), width=2)
    frac = float(np.clip(1.0 - dist_cm / 20.0, 0, 1))
    if frac > 0:
        d.rectangle([x0 + 2, y0 + 2, x0 + 2 + frac * (x1 - x0 - 4), y1 - 2], fill=col)
    if settled:
        d.text((x0, 58), "SETTLED", font=F_TAG, fill=(70, 214, 128, 255))
    elif inside:
        d.text((x0, 58), "IN TOLERANCE", font=F_TAG, fill=(238, 194, 66, 255))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default="media/rollout.mp4")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--no-dr", action="store_true")
    a = ap.parse_args()

    model, vn = load_policy(a.run)
    env = ServoArmReachEnv(reward_cfg=RewardCfg(),
                           dr_cfg=DomainRandCfg(enabled=not a.no_dr))
    renderer = mujoco.Renderer(env.model, a.height, a.width)
    cam = make_camera()
    thresh_cm = env.SUCCESS_RADIUS * 100

    frames, successes = [], 0
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=70000 + ep)
        info = dict(distance=np.linalg.norm(env._target - env._tip), inside=False, is_success=False)
        for _ in range(env.max_steps):
            o = vn.normalize_obs(obs.reshape(1, -1)) if vn is not None else obs.reshape(1, -1)
            act, _ = model.predict(o, deterministic=True)
            obs, _, _, trunc, info = env.step(np.asarray(act).reshape(-1))
            renderer.update_scene(env.data, camera=cam)
            frames.append(annotate(renderer.render(), ep, a.episodes,
                                   info["distance"] * 100, info["inside"],
                                   info["is_success"], thresh_cm))
            if trunc:
                break
        successes += int(info.get("is_success", 0))
        frames.extend([frames[-1]] * 10)      # hold, so episodes read separately

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    imageio.mimsave(a.out, frames, fps=20, quality=8, macro_block_size=1)
    print(f"wrote {a.out}: {len(frames)} frames, {a.episodes} episodes, "
          f"{successes}/{a.episodes} settled successfully")


if __name__ == "__main__":
    main()
