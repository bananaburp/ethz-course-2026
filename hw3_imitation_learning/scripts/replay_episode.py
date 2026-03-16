"""Replay a recorded teleop episode in the MuJoCo simulator.

Usage:
    python scripts/replay_episode.py --zarr <path_to.zarr> [--episode 0] [--speed 1.0]

Controls:
    SPACE   pause / resume
    n       next episode
    p       previous episode
    q/ESC   quit
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import warnings
import zarr

from hw3.teleop_utils import CAMERA_NAMES, JOINT_NAMES, compose_camera_views
from so101_gym.constants import ASSETS_DIR

warnings.filterwarnings("ignore")

XML_FALLBACK = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"


def load_episode(z: zarr.Group, ep_idx: int) -> dict[str, np.ndarray]:
    ends = z["meta/episode_ends"][:]
    start = int(ends[ep_idx - 1]) if ep_idx > 0 else 0
    end = int(ends[ep_idx])
    return {
        "state_joints": z["data/state_joints"][start:end],
        "state_cube": z["data/state_cube"][start:end],
        "state_obstacle": z["data/state_obstacle"][start:end],
    }


def replay(zarr_path: Path, ep_idx: int, speed: float) -> None:
    z = zarr.open(str(zarr_path))
    ends = z["meta/episode_ends"][:]
    n_episodes = len(ends)
    if n_episodes == 0:
        print("No episodes in this zarr file.")
        return

    # resolve XML — prefer the one stored in attrs, fall back to local
    xml_str: str = z.attrs.get("xml", "")
    xml_path = Path(xml_str) if xml_str and Path(xml_str).exists() else XML_FALLBACK
    print(f"Using XML: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)
    window = "Episode Replay"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    joint_names: list[str] = z.attrs.get("joint_names", list(JOINT_NAMES))
    qpos_idx = np.array(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in joint_names
        ],
        dtype=np.int32,
    )

    cube_joint_name = "red_box_joint"
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cube_joint_name)
    cube_qpos_idx = np.arange(
        model.jnt_qposadr[cube_jnt_id], model.jnt_qposadr[cube_jnt_id] + 7
    )

    obstacle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")

    dt = 1.0 / (z.attrs.get("control_hz", 10.0) * speed)

    paused = False
    current_ep = ep_idx % n_episodes

    while True:
        ep = load_episode(z, current_ep)
        n_frames = len(ep["state_joints"])
        print(f"Playing episode {current_ep + 1}/{n_episodes}  ({n_frames} frames)")

        frame = 0
        last = time.perf_counter()

        while frame < n_frames:
            k = cv2.waitKeyEx(1)
            if k in (ord("q"), 27):  # q or ESC
                cv2.destroyAllWindows()
                return
            if k == ord(" "):
                paused = not paused
            if k == ord("n"):
                current_ep = (current_ep + 1) % n_episodes
                break
            if k == ord("p"):
                current_ep = (current_ep - 1) % n_episodes
                break

            if paused:
                # still render last frame
                img = _render(renderer, data, model)
                img = _overlay(img, current_ep, n_episodes, frame, n_frames, paused)
                cv2.imshow(window, img)
                continue

            now = time.perf_counter()
            if now - last < dt:
                continue
            last = now

            # set joint positions
            data.qpos[qpos_idx] = ep["state_joints"][frame]
            # set cube pose
            data.qpos[cube_qpos_idx] = ep["state_cube"][frame]
            # set obstacle position
            if obstacle_body_id != -1:
                model.body_pos[obstacle_body_id] = ep["state_obstacle"][frame]
            mujoco.mj_forward(model, data)

            img = _render(renderer, data, model)
            img = _overlay(img, current_ep, n_episodes, frame, n_frames, paused)
            cv2.imshow(window, img)
            frame += 1

        else:
            # episode finished — auto-advance
            current_ep = (current_ep + 1) % n_episodes


def _render(
    renderer: mujoco.Renderer, data: mujoco.MjData, model: mujoco.MjModel
) -> np.ndarray:
    images = {}
    for cam in CAMERA_NAMES:
        renderer.update_scene(data, camera=cam)
        images[cam] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    return compose_camera_views(images, CAMERA_NAMES)


def _overlay(
    img: np.ndarray,
    ep_idx: int,
    n_eps: int,
    frame: int,
    n_frames: int,
    paused: bool,
) -> np.ndarray:
    img = img.copy()
    status = f"ep {ep_idx + 1}/{n_eps}  frame {frame}/{n_frames}  {'PAUSED' if paused else 'PLAYING'}"
    hint = "SPACE pause | n next | p prev | q quit"
    for text, y in ((status, 30), (hint, 60)):
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded teleop episode.")
    parser.add_argument("--zarr", type=Path, required=True, help="Path to .zarr file")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to start from (0-based)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 2.0 = 2x faster)")
    args = parser.parse_args()

    if not args.zarr.exists():
        raise FileNotFoundError(args.zarr)

    replay(args.zarr, args.episode, args.speed)


if __name__ == "__main__":
    main()
