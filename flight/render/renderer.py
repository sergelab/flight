from __future__ import annotations

import moderngl
import numpy as np

from flight.render.shaders import shader_sources
from flight.util.math import perspective

class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height

        vert, frag = shader_sources(ctx.version_code)
        self.prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        self.prog["u_proj"].write(perspective(70.0, width / height, 0.1, 800.0).tobytes())

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)  # safer default for early versions

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        self.prog["u_proj"].write(perspective(70.0, width / height, 0.1, 800.0).tobytes())

    def begin_frame(self) -> None:
        self.ctx.clear(0.70, 0.80, 0.92, 1.0)

    def set_common_uniforms(self, view: np.ndarray, cam_pos: np.ndarray, light_dir: np.ndarray, fog_start: float, fog_end: float) -> None:
        self.prog["u_view"].write(view.astype(np.float32).tobytes())
        self.prog["u_cam_pos"].value = (float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        self.prog["u_light_dir"].value = (float(light_dir[0]), float(light_dir[1]), float(light_dir[2]))
        self.prog["u_fog_start"].value = float(fog_start)
        self.prog["u_fog_end"].value = float(fog_end)

    def draw_chunk(self, vao: moderngl.VertexArray) -> None:
        vao.render()
