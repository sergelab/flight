from __future__ import annotations

from dataclasses import dataclass
import moderngl
import numpy as np

@dataclass
class ChunkCPU:
    cx: int
    cz: int
    vbo_data: np.ndarray  # interleaved float32 (N,6) -> flattened
    # indices shared globally, not stored here

@dataclass
class ChunkGPU:
    cx: int
    cz: int
    vao: moderngl.VertexArray
    vbo: moderngl.Buffer
