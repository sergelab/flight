from __future__ import annotations

from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Event
from typing import Callable, Iterable

import numpy as np

from flight.world.mesh_builder import build_chunk_vertices

@dataclass(frozen=True)
class CPUChunk:
    cx: int
    cz: int
    vbo_data: np.ndarray  # float32 (N,7) => pos(3), norm(3), water(1)
    ibo_data: np.ndarray  # uint32 (M,)
    # Tree instances for this chunk (variant C: real 3D trees)
    # Layout: x,y,z, scale, rot_y, kind, c0, c1 (8 floats)
    tree_instances: np.ndarray

class ChunkManager:
    def __init__(
        self,
        *,
        chunk_res: int,
        chunk_world_size: float,
        height_provider,
        skirts: bool = True,
        skirt_depth: float = 80.0,
        trees_enabled: bool = False,
        tree_density: float = 1.0,
    ) -> None:
        self.res = int(chunk_res)
        self.world_size = float(chunk_world_size)
        self.height_provider = height_provider
        self.skirts = bool(skirts)
        self.skirt_depth = float(skirt_depth)
        self.trees_enabled = bool(trees_enabled)
        self.tree_density = float(tree_density)

        self.in_q: "Queue[tuple[int,int]]" = Queue()
        self.out_q: "Queue[CPUChunk]" = Queue()

        self.pending: set[tuple[int,int]] = set()

        self._stop = Event()
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        # drain queue to unblock
        try:
            while True:
                self.in_q.get_nowait()
        except Exception:
            pass
        self._thread.join(timeout=1.0)

    def request(self, cx: int, cz: int) -> None:
        key = (int(cx), int(cz))
        if key in self.pending:
            return
        self.pending.add(key)
        self.in_q.put(key)

    def poll_ready(self, *, max_items: int = 16) -> list[CPUChunk]:
        ready: list[CPUChunk] = []
        for _ in range(int(max_items)):
            try:
                item = self.out_q.get_nowait()
            except Empty:
                break
            ready.append(item)
        return ready

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                cx, cz = self.in_q.get(timeout=0.05)
            except Empty:
                continue
            try:
                vbo, ibo, tree_instances = build_chunk_vertices(
                    cx,
                    cz,
                    self.res,
                    self.world_size,
                    self.height_provider,
                    skirts=self.skirts,
                    skirt_depth=self.skirt_depth,
                    trees_enabled=self.trees_enabled,
                    tree_density=self.tree_density,
                )
                self.out_q.put(CPUChunk(cx=cx, cz=cz, vbo_data=vbo, ibo_data=ibo, tree_instances=tree_instances))
            except Exception:
                # In production we'd log; for now just skip this chunk
                pass
