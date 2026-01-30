from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import moderngl
import numpy as np

from flight.world.chunk import ChunkCPU, ChunkGPU
from flight.world.chunk_manager import ChunkManager, ChunkWindow
from flight.world.mesh_builder import build_indices

@dataclass
class WorldParams:
    chunk_res: int
    chunk_world_size: float
    chunks_x: int
    chunks_z_behind: int
    chunks_z_ahead: int

class World:
    def __init__(self, ctx: moderngl.Context, params: WorldParams, height_provider) -> None:
        self.ctx = ctx
        self.params = params
        self.height_provider = height_provider

        self.indices = build_indices(params.chunk_res)
        self.ibo = ctx.buffer(self.indices.tobytes())

        self.window = ChunkWindow(params.chunks_x, params.chunks_z_behind, params.chunks_z_ahead)
        self.cm = ChunkManager(
            res=params.chunk_res,
            world_size=params.chunk_world_size,
            window=self.window,
            height_provider=height_provider,
        )

        self.chunks: Dict[Tuple[int,int], ChunkGPU] = {}


    def warmup(self, prog, *, min_chunks: int = 24, timeout_s: float = 2.0) -> None:
        """Preload some chunks into GPU so the first frame isn't empty.

        Blocks briefly (<= timeout_s) while uploading chunks as they are generated.
        """
        import time as _time
        deadline = _time.perf_counter() + float(timeout_s)
        while len(self.chunks) < int(min_chunks) and _time.perf_counter() < deadline:
            ready = self.cm.poll_ready(max_items=1)
            if not ready:
                _time.sleep(0.01)
                continue
            for cpu in ready:
                key = (cpu.cx, cpu.cz)
                if key in self.chunks:
                    continue
                vbo = self.ctx.buffer(cpu.vbo_data.tobytes())
                vao = self.ctx.vertex_array(
                    prog,
                    [
                        (vbo, "3f 3f", "in_pos", "in_norm"),
                    ],
                    self.ibo,
                )
                self.chunks[key] = ChunkGPU(cx=cpu.cx, cz=cpu.cz, vao=vao, vbo=vbo)

    def shutdown(self) -> None:
        self.cm.shutdown()
        for ch in self.chunks.values():
            try:
                ch.vao.release()
                ch.vbo.release()
            except Exception:
                pass
        try:
            self.ibo.release()
        except Exception:
            pass

    def update_requests(self, cam_x: float, cam_z: float) -> None:
        cam_cx, cam_cz = self.cm.world_to_chunk(cam_x, cam_z)
        needed = self.cm.needed_chunks(cam_cx, cam_cz)
        existing = set(self.chunks.keys())

        # Evict chunks outside window
        for key in list(self.chunks.keys()):
            if key not in needed:
                ch = self.chunks.pop(key)
                ch.vao.release()
                ch.vbo.release()

        # Request missing
        self.cm.request_missing(needed, set(self.chunks.keys()))

    def ingest_ready(self, prog, max_per_frame: int = 2) -> None:
        ready = self.cm.poll_ready(max_items=max_per_frame)
        for cpu in ready:
            key = (cpu.cx, cpu.cz)
            if key in self.chunks:
                continue

            vbo = self.ctx.buffer(cpu.vbo_data.tobytes())
            vao = self.ctx.vertex_array(
                prog,
                [
                    (vbo, "3f 3f", "in_pos", "in_norm"),
                ],
                self.ibo,
            )
            self.chunks[key] = ChunkGPU(cx=cpu.cx, cz=cpu.cz, vao=vao, vbo=vbo)

    def draw(self, renderer) -> None:
        for ch in self.chunks.values():
            renderer.draw_chunk(ch.vao)
