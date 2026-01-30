from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Dict, Tuple, Set

import numpy as np

from flight.world.chunk import ChunkCPU
from flight.world.mesh_builder import build_chunk_vertices

@dataclass(frozen=True)
class ChunkWindow:
    chunks_x: int
    z_behind: int
    z_ahead: int

class ChunkWorker(threading.Thread):
    def __init__(self, task_q: "queue.Queue[tuple[int,int]]", out_q: "queue.Queue[ChunkCPU]", *, res: int, world_size: float, height_fn) -> None:
        super().__init__(daemon=True)
        self.task_q = task_q
        self.out_q = out_q
        self.res = int(res)
        self.world_size = float(world_size)
        self.height_fn = height_fn
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                cx, cz = self.task_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                pos, norm = build_chunk_vertices(cx, cz, self.res, self.world_size, self.height_fn)
                interleaved = np.concatenate([pos, norm], axis=1).astype(np.float32).reshape(-1)
                self.out_q.put(ChunkCPU(cx=cx, cz=cz, vbo_data=interleaved))
            finally:
                self.task_q.task_done()

class ChunkManager:
    def __init__(self, *, res: int, world_size: float, window: ChunkWindow, height_fn) -> None:
        self.res = int(res)
        self.world_size = float(world_size)
        self.window = window
        self.height_fn = height_fn

        self.task_q: "queue.Queue[tuple[int,int]]" = queue.Queue()
        self.out_q: "queue.Queue[ChunkCPU]" = queue.Queue()

        self.worker = ChunkWorker(self.task_q, self.out_q, res=self.res, world_size=self.world_size, height_fn=self.height_fn)
        self.worker.start()

        self.pending: Set[Tuple[int,int]] = set()

    def shutdown(self) -> None:
        self.worker.stop()
        self.worker.join(timeout=1.0)

    def world_to_chunk(self, x: float, z: float) -> tuple[int,int]:
        cx = int(np.floor(x / self.world_size))
        cz = int(np.floor(z / self.world_size))
        return cx, cz

    def needed_chunks(self, cam_cx: int, cam_cz: int) -> Set[Tuple[int,int]]:
        half = self.window.chunks_x // 2
        needed: Set[Tuple[int,int]] = set()
        for dx in range(-half, half + 1):
            for dz in range(-self.window.z_behind, self.window.z_ahead + 1):
                needed.add((cam_cx + dx, cam_cz + dz))
        return needed

    def request_missing(self, needed: Set[Tuple[int,int]], existing: Set[Tuple[int,int]]) -> None:
        for key in needed:
            if key in existing or key in self.pending:
                continue
            self.pending.add(key)
            self.task_q.put(key)

    def poll_ready(self, max_items: int = 2) -> list[ChunkCPU]:
        ready = []
        for _ in range(max_items):
            try:
                ch = self.out_q.get_nowait()
            except queue.Empty:
                break
            self.pending.discard((ch.cx, ch.cz))
            ready.append(ch)
        return ready
