from __future__ import annotations

import numpy as np


def tree_shader_sources(glsl_version: int) -> tuple[str, str]:
    """Simple instanced tree shader (no textures, lowpoly).

    Instance layout: in_i_pos(3), in_i_scale(1), in_i_rot(1), in_i_kind(1), in_i_c0(1), in_i_c1(1)
    (we pack as 8 floats: pos3 + scale + rot + kind + c0 + c1)
    """
    prefix = f"#version {glsl_version}\n"

    vert = prefix + """
in vec3 in_pos;
in vec3 in_norm;

in vec3 in_i_pos;
in float in_i_scale;
in float in_i_rot;
in float in_i_kind;
in float in_i_c0;
in float in_i_c1;

uniform mat4 u_proj;
uniform mat4 u_view;

out vec3 v_world_pos;
out vec3 v_norm;
out float v_kind;
out vec2 v_c;

mat3 rot_y(float a) {
    float c = cos(a);
    float s = sin(a);
    return mat3(
        c, 0.0, -s,
        0.0, 1.0, 0.0,
        s, 0.0, c
    );
}

void main() {
    mat3 r = rot_y(in_i_rot);
    vec3 p = r * (in_pos * in_i_scale) + in_i_pos;
    vec3 n = normalize(r * in_norm);
    v_world_pos = p;
    v_norm = n;
    v_kind = in_i_kind;
    v_c = vec2(in_i_c0, in_i_c1);
    gl_Position = u_proj * u_view * vec4(p, 1.0);
}
"""

    frag = prefix + """
in vec3 v_world_pos;
in vec3 v_norm;
in float v_kind;
in vec2 v_c;

uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;
uniform float u_fog_start;
uniform float u_fog_end;

out vec4 f_color;

void main() {
    vec3 n = normalize(v_norm);
    vec3 l = normalize(u_light_dir);
    float diff = max(dot(n, l), 0.0);

    // Base palette: kind 0=pine, 1=deciduous.
    vec3 trunk = vec3(0.25, 0.17, 0.10);
    vec3 pine  = vec3(0.08, 0.26, 0.12);
    vec3 leaf  = vec3(0.10, 0.32, 0.10);
    vec3 foliage = mix(pine, leaf, clamp(v_kind, 0.0, 1.0));

    // Our mesh encodes trunk vs foliage by normal.y sign convention:
    // trunk normals have a small bias to be more vertical in generation.
    float trunk_mask = smoothstep(-0.05, 0.15, abs(n.y));
    vec3 base = mix(foliage, trunk, trunk_mask);

    // Subtle per-instance variation
    base *= vec3(v_c.x, v_c.y, 1.0);

    float ambient = 0.45;
    vec3 col = base * (ambient + 0.95 * diff);

    // Fog based on horizontal distance in any direction (XZ plane)
    vec2 d = v_world_pos.xz - u_cam_pos.xz;
    float dist = length(d);
    float fog_amount = smoothstep(u_fog_start, u_fog_end, dist);
    vec3 fog_col = vec3(0.70, 0.80, 0.92);
    col = mix(col, fog_col, fog_amount);

    f_color = vec4(col, 1.0);
}
"""

    return vert, frag


def build_tree_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return (vbo, ibo) for a small lowpoly tree.

    Vertex layout: pos(3), norm(3) float32.
    We generate one mesh that contains trunk + foliage:
      - trunk: a 6-sided prism
      - foliage: 2 stacked cones (very low poly)
    """
    verts: list[list[float]] = []
    norms: list[list[float]] = []
    idx: list[int] = []

    def add_tri(a, b, c):
        idx.extend([a, b, c])

    def add_vertex(p, n):
        verts.append([p[0], p[1], p[2]])
        norms.append([n[0], n[1], n[2]])
        return len(verts) - 1

    # --- trunk prism ---
    h0, h1 = 0.0, 2.2
    r = 0.25
    sides = 6
    for i in range(sides):
        a0 = (i / sides) * (2 * np.pi)
        a1 = ((i + 1) / sides) * (2 * np.pi)
        x0, z0 = float(np.cos(a0) * r), float(np.sin(a0) * r)
        x1, z1 = float(np.cos(a1) * r), float(np.sin(a1) * r)
        # Normal roughly outward
        nx, nz = float(np.cos((a0 + a1) * 0.5)), float(np.sin((a0 + a1) * 0.5))
        n = (nx, 0.15, nz)  # slight vertical bias to help trunk/foliage mix

        v00 = add_vertex((x0, h0, z0), n)
        v01 = add_vertex((x1, h0, z1), n)
        v10 = add_vertex((x0, h1, z0), n)
        v11 = add_vertex((x1, h1, z1), n)
        add_tri(v00, v01, v10)
        add_tri(v01, v11, v10)

    # --- foliage cones ---
    def add_cone(y_base: float, y_tip: float, radius: float, sides: int = 7):
        tip = add_vertex((0.0, y_tip, 0.0), (0.0, 1.0, 0.0))
        ring: list[int] = []
        for i in range(sides):
            a = (i / sides) * (2 * np.pi)
            x, z = float(np.cos(a) * radius), float(np.sin(a) * radius)
            # normal points halfway between side and up
            n = (x, radius * 0.65, z)
            ring.append(add_vertex((x, y_base, z), n))
        for i in range(sides):
            a = ring[i]
            b = ring[(i + 1) % sides]
            add_tri(a, b, tip)

    add_cone(1.6, 4.2, 1.25, sides=7)
    add_cone(2.4, 5.3, 0.95, sides=7)

    vbo = np.concatenate([np.array(verts, dtype=np.float32), np.array(norms, dtype=np.float32)], axis=1).astype(np.float32)
    ibo = np.array(idx, dtype=np.uint32)
    return vbo, ibo
