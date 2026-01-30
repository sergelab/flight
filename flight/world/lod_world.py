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
        self.near.warmup(prog, min_chunks=24, timeout_s=timeout_s)
        self.far.warmup(prog, min_chunks=12, timeout_s=min(1.0, timeout_s))

    def update_requests(self, x: float, z: float) -> None:
        self.near.update_requests(x, z)
        if (self._frame % max(1, self.params.far_update_every_n_frames)) == 0:
            self.far.update_requests(x, z)
        self._frame += 1

    def ingest_ready(self, prog, *, max_per_frame: int) -> None:
        # B: still prioritize near, but keep far moving
        near_budget = max(1, int(max_per_frame * 0.70))
        far_budget = max(0, int(max_per_frame - near_budget))
        self.near.ingest_ready(prog, max_per_frame=near_budget)
        if far_budget > 0:
            self.far.ingest_ready(prog, max_per_frame=far_budget)


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
        self.near.warmup(prog, min_chunks=24, timeout_s=timeout_s)
        self.far.warmup(prog, min_chunks=12, timeout_s=min(1.0, timeout_s))

    def update_requests(self, x: float, z: float) -> None:
        self.near.update_requests(x, z)
        if (self._frame % max(1, self.params.far_update_every_n_frames)) == 0:
            self.far.update_requests(x, z)
        self._frame += 1

    def ingest_ready(self, prog, *, max_per_frame: int) -> None:
        near_budget = max(1, int(max_per_frame * 0.70))
        far_budget = max(0, int(max_per_frame - near_budget))
        self.near.ingest_ready(prog, max_per_frame=near_budget)
        if far_budget > 0:
            self.far.ingest_ready(prog, max_per_frame=far_budget)

    def draw(self, renderer, *, cam_z: float, chunk_size: float) -> None:
        # v0.3.2 B: reduce flicker (z-fighting) and make fades correct
        # Overlap band where both rings exist; we fade near out and far in.
        blend_start = 3.0 * chunk_size
        blend_end = 10.0 * chunk_size

        # Draw FAR first with depth bias to reduce z-fighting
        if hasattr(renderer, "begin_far"):
            renderer.begin_far()
        self.far.draw(renderer, cam_z=cam_z, fade_from=blend_start, fade_to=blend_end, invert=False)
        if hasattr(renderer, "end_far"):
            renderer.end_far()

        # Draw NEAR after: fully visible close, fade out after blend_start towards blend_end
        self.near.draw(renderer, cam_z=cam_z, fade_from=blend_start, fade_to=blend_end, invert=True)
