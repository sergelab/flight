from __future__ import annotations

import moderngl
import numpy as np

from flight.render.shaders import shader_sources
from flight.util.math import perspective

_DEBUG_VERT = """#version 150
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_DEBUG_FRAG = """#version 150
out vec4 f_color;
void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height

        vert, frag = shader_sources(ctx.version_code)
        self.prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        self.prog["u_proj"].write(perspective(70.0, width / height, 0.1, 800.0).tobytes())

        self.ctx.enable(moderngl.DEPTH_TEST)
        # Safer early: no culling until visuals confirmed on target GPUs
        self.ctx.disable(moderngl.CULL_FACE)

        # Debug triangle in NDC (bottom-left). Uses a tiny GLSL 150 program (works on GL 3.2+).
        self._dbg_prog = self.ctx.program(vertex_shader=_DEBUG_VERT, fragment_shader=_DEBUG_FRAG)
        dbg_vertices = np.array([
            -0.95, -0.95,
            -0.75, -0.95,
            -0.95, -0.75,
        ], dtype=np.float32)
        self._dbg_vbo = self.ctx.buffer(dbg_vertices.tobytes())
        self._dbg_vao = self.ctx.vertex_array(self._dbg_prog, [(self._dbg_vbo, "2f", "in_pos")])

    def release(self) -> None:
        try:
            self._dbg_vao.release()
            self._dbg_vbo.release()
            self._dbg_prog.release()
            self.prog.release()
        except Exception:
            pass

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

    def draw_debug(self) -> None:
        # Draw after terrain; disable depth so it always shows.
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._dbg_vao.render(mode=moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
