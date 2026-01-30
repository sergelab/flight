from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import moderngl

from flight.world.chunk_manager import ChunkManager, CPUChunk

@dataclass(frozen=True)
class WorldParams:
    chunk_res: int
    chunk_world_size: float
    chunks_x: int
    chunks_z_behind: int
    chunks_z_ahead: int
    skirts: bool = True
    skirt_depth: float = 80.0

@dataclass
class ChunkGPU:
    cx: int
    cz: int
    vao: moderngl.VertexArray
    vbo: moderngl.Buffer

class World:
    def __init__(self, ctx: moderngl.Context, params: WorldParams, height_provider) -> None:
        self.ctx = ctx
        self.params = params
        self.height_provider = height_provider

        self.cm = ChunkManager(
            chunk_res=params.chunk_res,
            chunk_world_size=params.chunk_world_size,
            height_provider=height_provider,
            skirts=params.skirts,
            skirt_depth=params.skirt_depth,
        )

        self.chunks: Dict[Tuple[int,int], ChunkGPU] = {}
        self._ibo: moderngl.Buffer | None = None
        self._ibo_len = 0

        self._last_center = (None, None)

    def shutdown(self) -> None:
        self.cm.shutdown()
        for c in list(self.chunks.values()):
            try:
                c.vao.release()
                c.vbo.release()
            except Exception:
                pass
        self.chunks.clear()
        try:
            if self._ibo is not None:
                self._ibo.release()
        except Exception:
            pass

    def update_requests(self, x: float, z: float) -> None:
        # Determine chunk center
        cx0 = int(x // self.params.chunk_world_size)
        cz0 = int(z // self.params.chunk_world_size)

        if self._last_center == (cx0, cz0):
            return
        self._last_center = (cx0, cz0)

        half = self.params.chunks_x // 2
        for dx in range(-half, half + 1):
            for dz in range(-self.params.chunks_z_behind, self.params.chunks_z_ahead + 1):
                cx = cx0 + dx
                cz = cz0 + dz
                key = (cx, cz)
                if key in self.chunks:
                    continue
                self.cm.request(cx, cz)

        # Optional: could evict far-behind chunks to save VRAM (not aggressive yet)

    def warmup(self, prog, *, min_chunks: int = 24, timeout_s: float = 2.0) -> None:
        import time as _time
        deadline = _time.perf_counter() + float(timeout_s)
        while len(self.chunks) < int(min_chunks) and _time.perf_counter() < deadline:
            self.ingest_ready(prog, max_per_frame=32)
            if len(self.chunks) >= int(min_chunks):
                break
            _time.sleep(0.01)

    def ingest_ready(self, prog, *, max_per_frame: int = 8) -> None:
        ready = self.cm.poll_ready(max_items=int(max_per_frame))
        for cpu in ready:
            key = (cpu.cx, cpu.cz)
            self.cm.pending.discard(key)
            if key in self.chunks:
                continue

            if self._ibo is None:
                self._ibo = self.ctx.buffer(cpu.ibo_data.tobytes())
                self._ibo_len = int(cpu.ibo_data.size)

            vbo = self.ctx.buffer(cpu.vbo_data.tobytes())
            vao = self.ctx.vertex_array(
                prog,
                [
                    (vbo, "3f 3f 1f", "in_pos", "in_norm", "in_water"),
                ],
                self._ibo,
            )
            self.chunks[key] = ChunkGPU(cx=cpu.cx, cz=cpu.cz, vao=vao, vbo=vbo)

    def draw(self, renderer) -> None:
        # Simple draw order: just draw all
        for c in self.chunks.values():
            renderer.draw_chunk(c.vao)
