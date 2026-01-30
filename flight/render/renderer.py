from __future__ import annotations

import moderngl
import numpy as np

from flight.render.shaders import shader_sources
from flight.util.math import perspective

_HUD_VERT = """#version 150
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_HUD_FRAG = """#version 150
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_tex, v_uv);
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
        self.ctx.disable(moderngl.CULL_FACE)  # safe default for now

        # HUD quad (top-left)
        self._hud_prog = self.ctx.program(vertex_shader=_HUD_VERT, fragment_shader=_HUD_FRAG)
        quad = np.array([
            # x, y, u, v  (top-left box)
            -0.98,  0.98, 0.0, 1.0,
            -0.20,  0.98, 1.0, 1.0,
            -0.98,  0.70, 0.0, 0.0,

            -0.20,  0.98, 1.0, 1.0,
            -0.20,  0.70, 1.0, 0.0,
            -0.98,  0.70, 0.0, 0.0,
        ], dtype=np.float32)
        self._hud_vbo = self.ctx.buffer(quad.tobytes())
        self._hud_vao = self.ctx.vertex_array(self._hud_prog, [(self._hud_vbo, "2f 2f", "in_pos", "in_uv")])

        self._hud_tex: moderngl.Texture | None = None
        self._hud_tex_size = (0, 0)

    def release(self) -> None:
        for obj in [self._hud_vao, self._hud_vbo, self._hud_prog, self.prog]:
            try:
                obj.release()
            except Exception:
                pass
        try:
            if self._hud_tex is not None:
                self._hud_tex.release()
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
        if "u_chunk_fade" in self.prog:
            self.prog["u_chunk_fade"].value = 1.0

    def begin_far(self) -> None:
        # v0.3.3: eliminate LOD flicker by preventing FAR ring from writing depth.
        # FAR is drawn first as background; NEAR draws over it where available.
        try:
            self.ctx.depth_mask = False
            self.ctx.enable(moderngl.POLYGON_OFFSET_FILL)
            self.ctx.polygon_offset = (1.0, 2.0)
        except Exception:
            pass

    def end_far(self) -> None:
        try:
            self.ctx.disable(moderngl.POLYGON_OFFSET_FILL)
        except Exception:
            pass
        try:
            self.ctx.depth_mask = True
        except Exception:
            pass

    def set_chunk_fade(self, fade: float) -> None:
        if "u_chunk_fade" in self.prog:
            self.prog["u_chunk_fade"].value = float(fade)

    def draw_chunk(self, vao: moderngl.VertexArray) -> None:
        vao.render()

    # --- HUD ---
    def hud_update_rgba(self, rgba_bytes: bytes, w: int, h: int) -> None:
        if self._hud_tex is None or self._hud_tex_size != (w, h):
            if self._hud_tex is not None:
                self._hud_tex.release()
            self._hud_tex = self.ctx.texture((w, h), 4, data=rgba_bytes)
            self._hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._hud_tex.repeat_x = False
            self._hud_tex.repeat_y = False
            self._hud_tex_size = (w, h)
        else:
            self._hud_tex.write(rgba_bytes)

    def draw_hud(self) -> None:
        if self._hud_tex is None:
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._hud_tex.use(location=0)
        self._hud_prog["u_tex"].value = 0
        self._hud_vao.render(mode=moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
