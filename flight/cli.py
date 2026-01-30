from __future__ import annotations

import argparse
import random

from flight.app import run_app
from flight.config import (
    DEFAULT_HEIGHT_OFFSET,
    DEFAULT_SEED,
    DEFAULT_SPEED,
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
    p = argparse.ArgumentParser(prog="flight", description="Infinite flight over mountains (ModernGL + pygame)")
    p.add_argument("--seed", default=str(DEFAULT_SEED), help="int seed or 'random' (default: 12345)")
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="forward speed (world units / sec)")
    p.add_argument("--height-offset", type=float, default=DEFAULT_HEIGHT_OFFSET, help="camera height above terrain")
    p.add_argument("--wireframe", action="store_true", help="render wireframe")
    p.add_argument("--chunk-res", type=int, default=DEFAULT_CHUNK_RES, help="vertices per chunk side (default: 64)")
    p.add_argument("--chunk-size", type=float, default=DEFAULT_CHUNK_WORLD_SIZE, help="chunk world size (default: 64.0)")
    p.add_argument("--fog-start", type=float, default=DEFAULT_FOG_START, help="fog start distance")
    p.add_argument("--fog-end", type=float, default=DEFAULT_FOG_END, help="fog end distance")
    p.add_argument("--debug", action="store_true", help="enable debug overlay (HUD + logs)")
    p.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS, help="target FPS for adaptive streaming")
    p.add_argument("--lod", dest="lod", action="store_true", default=DEFAULT_LOD, help="enable LOD rings (default off)")
    p.add_argument("--no-lod", dest="lod", action="store_false", help="disable LOD rings")
    p.add_argument("--noise", choices=["fast","simplex"], default=DEFAULT_NOISE, help="height noise mode (fast or simplex)")
    p.add_argument("--trees", dest="trees", action="store_true", default=DEFAULT_TREES, help="enable forests (3D trees) (default on)")
    p.add_argument("--no-trees", dest="trees", action="store_false", help="disable forests (3D trees)")
    p.add_argument("--tree-density", type=float, default=DEFAULT_TREE_DENSITY, help="tree density multiplier (default 1.0)")
    p.add_argument("--auto", action="store_true", default=DEFAULT_AUTO, help="auto forward flight (like previous versions)")
    p.add_argument("--turn-rate", type=float, default=DEFAULT_TURN_RATE, help="turn rate in radians/sec for manual flight")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    if isinstance(args.seed, str) and args.seed.lower() == "random":
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(args.seed)

    run_app(
        seed=seed,
        speed=float(args.speed),
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
    )
