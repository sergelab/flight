from __future__ import annotations

from dataclasses import dataclass

import moderngl

from flight.world.world import World, WorldParams

@dataclass(frozen=True)
class LODParams:
    near: WorldParams
    far: WorldParams
    far_update_every_n_frames: int = 2

class LODWorld:
    def __init__(self, ctx: moderngl.Context, params: LODParams, height_provider) -> None:
        self.ctx = ctx
        self.params = params
        self.height_provider = height_provider

        self.near = World(ctx, params.near, height_provider)
        self.far = World(ctx, params.far, height_provider)

        self._frame = 0

    def shutdown(self) -> None:
        self.near.shutdown()
        self.far.shutdown()

    def warmup(self, prog, *, timeout_s: float = 2.0) -> None:
        # Warm up near more than far (A: FPS priority)
        self.near.warmup(prog, min_chunks=28, timeout_s=timeout_s)
        self.far.warmup(prog, min_chunks=12, timeout_s=min(1.0, timeout_s))

    def update_requests(self, x: float, z: float) -> None:
        self.near.update_requests(x, z)
        if (self._frame % max(1, self.params.far_update_every_n_frames)) == 0:
            self.far.update_requests(x, z)
        self._frame += 1

    def ingest_ready(self, prog, *, max_per_frame: int) -> None:
        # A: prioritize near uploads
        near_budget = max(1, int(max_per_frame * 0.75))
        far_budget = max(0, int(max_per_frame - near_budget))
        self.near.ingest_ready(prog, max_per_frame=near_budget)
        if far_budget > 0:
            self.far.ingest_ready(prog, max_per_frame=far_budget)

    def draw(self, renderer) -> None:
        # Draw far first then near
        self.far.draw(renderer)
        self.near.draw(renderer)
