from __future__ import annotations

import argparse
import random

from flight.app import run_app
from flight.config import DEFAULT_HEIGHT_OFFSET, DEFAULT_SEED, DEFAULT_SPEED

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="flight", description="Infinite flight over mountains (ModernGL + pygame)")
    p.add_argument("--seed", default=str(DEFAULT_SEED), help="int seed or 'random' (default: 12345)")
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="forward speed (world units / sec)")
    p.add_argument("--height-offset", type=float, default=DEFAULT_HEIGHT_OFFSET, help="camera height above terrain")
    p.add_argument("--wireframe", action="store_true", help="render wireframe")
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
    )
