from __future__ import annotations

import moderngl
import numpy as np

from flight.render.shaders import shader_sources
from flight.render.trees import tree_shader_sources, build_tree_mesh
from flight.util.math import perspective

_SKY_VERT = """#version 150
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_SKY_FRAG = """#version 150
in vec2 v_uv;
uniform vec2 u_sun_ndc;
out vec4 f_color;

void main() {
    // Gradient: horizon -> zenith
    vec3 horizon = vec3(0.78, 0.86, 0.96);
    vec3 zenith  = vec3(0.40, 0.60, 0.85);
    float t = smoothstep(0.0, 1.0, v_uv.y);
    vec3 col = mix(horizon, zenith, t);

    // Sun disc (in NDC)
    vec2 ndc = vec2(v_uv.x * 2.0 - 1.0, v_uv.y * 2.0 - 1.0);
    float d = length(ndc - u_sun_ndc);
    float disc = 1.0 - smoothstep(0.05, 0.08, d);
    float glow = 1.0 - smoothstep(0.10, 0.35, d);
    col += vec3(1.0, 0.95, 0.80) * (disc * 0.75 + glow * 0.12);

    f_color = vec4(col, 1.0);
}
"""

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

        # Trees (variant C): separate instanced shader.
        # We keep this optional: if something fails to compile on a given GPU, the app still runs.
        self.tree_prog: moderngl.Program | None = None
        self._tree_vbo: moderngl.Buffer | None = None
        self._tree_ibo: moderngl.Buffer | None = None
        self._tree_ibo_len: int = 0
        try:
            glsl = 330 if ctx.version_code >= 330 else 150
            tvert, tfrag = tree_shader_sources(glsl)
            self.tree_prog = self.ctx.program(vertex_shader=tvert, fragment_shader=tfrag)
            tvbo, tibo = build_tree_mesh()
            self._tree_vbo = self.ctx.buffer(tvbo.tobytes())
            self._tree_ibo = self.ctx.buffer(tibo.tobytes())
            self._tree_ibo_len = int(tibo.size)
            # Static uniforms
            self.tree_prog["u_proj"].write(perspective(70.0, width / height, 0.1, 800.0).astype(np.float32).tobytes())
        except Exception:
            self.tree_prog = None

        self._proj = perspective(70.0, width / height, 0.1, 800.0).astype(np.float32)
        self.prog["u_proj"].write(self._proj.tobytes())
        if "u_chunk_fade" in self.prog:
            self.prog["u_chunk_fade"].value = 1.0

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # Sky quad
        self._sky_prog = self.ctx.program(vertex_shader=_SKY_VERT, fragment_shader=_SKY_FRAG)
        sky = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)
        self._sky_vbo = self.ctx.buffer(sky.tobytes())
        self._sky_vao = self.ctx.vertex_array(self._sky_prog, [(self._sky_vbo, "2f", "in_pos")])

        # HUD quad (top-left)
        self._hud_prog = self.ctx.program(vertex_shader=_HUD_VERT, fragment_shader=_HUD_FRAG)
        quad = np.array([
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

    @property
    def proj(self) -> np.ndarray:
        return self._proj

    def release(self) -> None:
        for obj in [self._sky_vao, self._sky_vbo, self._sky_prog, self._hud_vao, self._hud_vbo, self._hud_prog, self.prog]:
            try:
                obj.release()
            except Exception:
                pass

        try:
            if self._hud_tex is not None:
                self._hud_tex.release()
        except Exception:
            pass

        for obj in [self.tree_prog, self._tree_vbo, self._tree_ibo]:
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        self._proj = perspective(70.0, width / height, 0.1, 800.0).astype(np.float32)
        self.prog["u_proj"].write(self._proj.tobytes())
        if self.tree_prog is not None and "u_proj" in self.tree_prog:
            self.tree_prog["u_proj"].write(self._proj.tobytes())
        if "u_chunk_fade" in self.prog:
            self.prog["u_chunk_fade"].value = 1.0

    def begin_frame(self) -> None:
        # Clear is mostly hidden by sky quad, but keep for safety
        self.ctx.clear(0.70, 0.80, 0.92, 1.0)

    def draw_sky(self, sun_ndc: tuple[float, float]) -> None:
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._sky_prog["u_sun_ndc"].value = (float(sun_ndc[0]), float(sun_ndc[1]))
        self._sky_vao.render(mode=moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def set_common_uniforms(
        self,
        view: np.ndarray,
        cam_pos: np.ndarray,
        light_dir: np.ndarray,
        fog_start: float,
        fog_end: float,
        *,
        time_s: float = 0.0,
    ) -> None:
        self.prog["u_view"].write(view.astype(np.float32).tobytes())
        self.prog["u_cam_pos"].value = (float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        self.prog["u_light_dir"].value = (float(light_dir[0]), float(light_dir[1]), float(light_dir[2]))
        if "u_time" in self.prog:
            self.prog["u_time"].value = float(time_s)
        self.prog["u_fog_start"].value = float(fog_start)
        self.prog["u_fog_end"].value = float(fog_end)
        if "u_chunk_fade" in self.prog:
            self.prog["u_chunk_fade"].value = 1.0

        # Trees share most uniforms with terrain (no time needed here)
        if self.tree_prog is not None:
            self.tree_prog["u_view"].write(view.astype(np.float32).tobytes())
            self.tree_prog["u_cam_pos"].value = (float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
            self.tree_prog["u_light_dir"].value = (float(light_dir[0]), float(light_dir[1]), float(light_dir[2]))
            self.tree_prog["u_fog_start"].value = float(fog_start)
            self.tree_prog["u_fog_end"].value = float(fog_end)

        # (proj is updated on resize)

    def set_chunk_fade(self, fade: float) -> None:
        if "u_chunk_fade" in self.prog:
            self.prog["u_chunk_fade"].value = float(fade)

    def draw_chunk(self, vao: moderngl.VertexArray) -> None:
            vao.render()

    # --- Trees ---
    def make_tree_vao(self, instance_vbo: moderngl.Buffer) -> moderngl.VertexArray:
        if self.tree_prog is None or self._tree_vbo is None or self._tree_ibo is None:
            raise RuntimeError("Tree renderer is not initialized")
        # Base mesh: pos+norm, per-vertex
        # Instance buffer: 8 floats per instance
        return self.ctx.vertex_array(
            self.tree_prog,
            [
                (self._tree_vbo, "3f 3f", "in_pos", "in_norm"),
                (instance_vbo, "3f 1f 1f 1f 1f 1f/i", "in_i_pos", "in_i_scale", "in_i_rot", "in_i_kind", "in_i_c0", "in_i_c1"),
            ],
            self._tree_ibo,
        )

    def draw_trees(self, vao: moderngl.VertexArray, instance_count: int) -> None:
        # Instance rendering
        vao.render(instances=int(instance_count))

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
