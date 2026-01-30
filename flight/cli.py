from __future__ import annotations

import argparse
import random

from flight.app import run_app
from flight.config import DEFAULT_HEIGHT_OFFSET, DEFAULT_SEED, DEFAULT_SPEED, DEFAULT_CHUNK_RES, DEFAULT_CHUNK_WORLD_SIZE, DEFAULT_FOG_START, DEFAULT_FOG_END

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
    p.add_argument("--noise", choices=["fast","simplex"], default="fast", help="height noise mode (fast or simplex)")
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
        noise_mode=str(args.noise),
        chunk_res=int(args.chunk_res),
        chunk_size=float(args.chunk_size),
        fog_start=float(args.fog_start),
        fog_end=float(args.fog_end),
    )
