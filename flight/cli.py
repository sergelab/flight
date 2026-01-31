from __future__ import annotations

import argparse
import random

from flight.app import run_app
from flight.config import (
    APP_VERSION,
    DEFAULT_HEIGHT_OFFSET,
    DEFAULT_SEED,
    DEFAULT_SPEED,
    DEFAULT_MAX_SPEED,
    DEFAULT_ACCEL,
    DEFAULT_BRAKE,
    DEFAULT_DRAG,
    DEFAULT_MAX_YAW_RATE_SLOW,
    DEFAULT_MAX_YAW_RATE_FAST,
    DEFAULT_YAW_ACCEL,
    DEFAULT_YAW_DECEL,
    DEFAULT_YAW_DRAG,
    DEFAULT_BANK_GAIN,
    DEFAULT_BANK_MAX,
    DEFAULT_BANK_SMOOTH_K,
    DEFAULT_INPUT_SMOOTH_K,
    DEFAULT_CAM_YAW_SMOOTH_K,
    DEFAULT_PITCH_GAIN,
    DEFAULT_PITCH_MAX,
    DEFAULT_PITCH_SMOOTH_K,
    DEFAULT_CHUNK_RES,
    DEFAULT_CHUNK_WORLD_SIZE,
    DEFAULT_FOG_START,
    DEFAULT_FOG_END,
    DEFAULT_TARGET_FPS,
    DEFAULT_LOD,
    DEFAULT_NOISE,
    DEFAULT_TREES,
    DEFAULT_TREE_DENSITY,
    DEFAULT_AUTO,
    DEFAULT_TURN_RATE,
)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="flight", description=f"Infinite flight over mountains (ModernGL + pygame) v{APP_VERSION}")
    p.add_argument("--seed", default=str(DEFAULT_SEED), help="int seed or 'random' (default: 12345)")
    p.add_argument("--speed", type=float, default=DEFAULT_MAX_SPEED, help="max forward speed (legacy name: speed) (world units / sec)")
    p.add_argument("--accel", type=float, default=DEFAULT_ACCEL, help="acceleration toward target speed (units/sec^2)")
    p.add_argument("--brake", type=float, default=DEFAULT_BRAKE, help="deceleration toward target speed (units/sec^2)")
    p.add_argument("--drag", type=float, default=DEFAULT_DRAG, help="speed drag (1/sec), higher = faster coasting decay")
    p.add_argument("--height-offset", type=float, default=DEFAULT_HEIGHT_OFFSET, help="camera height above terrain")
    p.add_argument("--wireframe", action="store_true", help="render wireframe")
    p.add_argument("--chunk-res", type=int, default=DEFAULT_CHUNK_RES, help="vertices per chunk side (default: 64)")
    p.add_argument("--chunk-size", type=float, default=DEFAULT_CHUNK_WORLD_SIZE, help="chunk world size (default: 64.0)")
    p.add_argument("--fog-start", type=float, default=DEFAULT_FOG_START, help="fog start distance")
    p.add_argument("--fog-end", type=float, default=DEFAULT_FOG_END, help="fog end distance")
    p.add_argument("--debug", action="store_true", help="enable debug overlay (HUD + logs)")
    p.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS, help="target FPS for adaptive streaming")
    p.add_argument(
        "--lod",
        dest="lod",
        action="store_true",
        default=DEFAULT_LOD,
        help="enable LOD rings (default off)",
    )
    p.add_argument("--no-lod", dest="lod", action="store_false", help="disable LOD rings")
    p.add_argument("--noise", choices=["fast","simplex"], default=DEFAULT_NOISE, help="height noise mode (fast or simplex)")
    p.add_argument("--trees", dest="trees", action="store_true", default=DEFAULT_TREES, help="enable forests (3D trees) (default on)")
    p.add_argument("--no-trees", dest="trees", action="store_false", help="disable forests (3D trees)")
    p.add_argument("--tree-density", type=float, default=DEFAULT_TREE_DENSITY, help="tree density multiplier (default 1.0)")
    p.add_argument(
        "--auto",
        action="store_true",
        default=DEFAULT_AUTO,
        help="auto-flight forward (no manual controls) (default off)",
    )
    p.add_argument(
        "--turn-rate",
        type=float,
        default=DEFAULT_TURN_RATE,
        help="legacy: direct yaw rate from v0.8 (rad/sec). In v0.9 used as fallback for yaw-rate caps.",
    )
    p.add_argument("--yaw-rate-slow", type=float, default=DEFAULT_MAX_YAW_RATE_SLOW, help="max yaw rate at low speed (rad/sec)")
    p.add_argument("--yaw-rate-fast", type=float, default=DEFAULT_MAX_YAW_RATE_FAST, help="max yaw rate at high speed (rad/sec)")
    p.add_argument("--yaw-accel", type=float, default=DEFAULT_YAW_ACCEL, help="yaw acceleration toward target yaw rate (rad/sec^2)")
    p.add_argument("--yaw-decel", type=float, default=DEFAULT_YAW_DECEL, help="yaw deceleration toward target yaw rate (rad/sec^2)")
    p.add_argument("--yaw-drag", type=float, default=DEFAULT_YAW_DRAG, help="yaw-rate drag (1/sec), higher = faster stop")
    p.add_argument("--bank-gain", type=float, default=DEFAULT_BANK_GAIN, help="bank response gain (rad bank per rad/sec yaw)")
    p.add_argument("--bank-max", type=float, default=DEFAULT_BANK_MAX, help="max bank angle (radians)")
    p.add_argument("--bank-smooth", type=float, default=DEFAULT_BANK_SMOOTH_K, help="bank smoothing (1/sec)")
    p.add_argument("--input-smooth", type=float, default=DEFAULT_INPUT_SMOOTH_K, help="smooth arrow inputs (1/sec), lower = heavier")
    p.add_argument("--cam-yaw-smooth", type=float, default=DEFAULT_CAM_YAW_SMOOTH_K, help="camera yaw lag smoothing (1/sec)")
    p.add_argument("--pitch-gain", type=float, default=DEFAULT_PITCH_GAIN, help="visual pitch per longitudinal accel (rad per (unit/sec^2))")
    p.add_argument("--pitch-max", type=float, default=DEFAULT_PITCH_MAX, help="max visual pitch (radians)")
    p.add_argument("--pitch-smooth", type=float, default=DEFAULT_PITCH_SMOOTH_K, help="visual pitch smoothing (1/sec)")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    if isinstance(args.seed, str) and args.seed.lower() == "random":
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(args.seed)

    run_app(
        seed=seed,
        max_speed=float(args.speed),
        accel=float(args.accel),
        brake=float(args.brake),
        drag=float(args.drag),
        height_offset=float(args.height_offset),
        wireframe=bool(args.wireframe),
        debug=bool(args.debug),
        target_fps=int(args.target_fps),
        lod=bool(args.lod),
        noise_mode=str(args.noise),
        chunk_res=int(args.chunk_res),
        chunk_size=float(args.chunk_size),
        fog_start=float(args.fog_start),
        fog_end=float(args.fog_end),
        trees=bool(args.trees),
        tree_density=float(args.tree_density),
        auto=bool(args.auto),
        turn_rate=float(args.turn_rate),
        yaw_rate_slow=float(args.yaw_rate_slow),
        yaw_rate_fast=float(args.yaw_rate_fast),
        yaw_accel=float(args.yaw_accel),
        yaw_decel=float(args.yaw_decel),
        yaw_drag=float(args.yaw_drag),
        bank_gain=float(args.bank_gain),
        bank_max=float(args.bank_max),
        bank_smooth_k=float(args.bank_smooth),
        input_smooth_k=float(args.input_smooth),
        cam_yaw_smooth_k=float(args.cam_yaw_smooth),
        pitch_gain=float(args.pitch_gain),
        pitch_max=float(args.pitch_max),
        pitch_smooth_k=float(args.pitch_smooth),
    )
