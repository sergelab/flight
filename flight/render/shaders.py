from __future__ import annotations

def _pick_glsl_version(ctx_version_code: int) -> int:
    """Pick a GLSL version compatible with the active OpenGL context.

    - For OpenGL >= 3.3: use GLSL 330
    - For OpenGL >= 3.2: use GLSL 150
    """
    if ctx_version_code >= 330:
        return 330
    if ctx_version_code >= 320:
        return 150
    # As a last resort, try 150; but the project targets modern contexts.
    return 150

_VERT_BODY = """
in vec3 in_pos;
in vec3 in_norm;

uniform mat4 u_proj;
uniform mat4 u_view;

out vec3 v_world_pos;
out vec3 v_norm;
out float v_height;

void main() {
    v_world_pos = in_pos;
    v_norm = in_norm;
    v_height = in_pos.y;
    gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
}
"""

_FRAG_BODY = """in vec3 v_world_pos;
in vec3 v_norm;
in float v_height;

uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;
uniform float u_fog_start;
uniform float u_fog_end;
uniform float u_chunk_fade;

out vec4 f_color;

vec3 height_color(float h) {
    // Simple palette: low->greenish, mid->gray, high->white
    float t1 = smoothstep(-10.0, 10.0, h);
    float t2 = smoothstep(10.0, 35.0, h);
    vec3 low = vec3(0.10, 0.35, 0.15);
    vec3 mid = vec3(0.35, 0.35, 0.35);
    vec3 high = vec3(0.85, 0.85, 0.90);
    vec3 a = mix(low, mid, t1);
    return mix(a, high, t2);
}

void main() {
    vec3 n = normalize(v_norm);
    vec3 l = normalize(u_light_dir);
    float diff = max(dot(n, l), 0.0);

    vec3 base = height_color(v_height);

    float dist = max(v_world_pos.z - u_cam_pos.z, 0.0);

    float detail = 1.0 - smoothstep(200.0, 650.0, dist);

    float amb = mix(0.44, 0.58, smoothstep(0.0, 900.0, dist));
    vec3 col = base * (amb + 0.95 * diff);

    float spacing = 7.0;
    float hh = v_height / spacing;
    float fracv = abs(fract(hh) - 0.5);
    float aa = max(fwidth(hh), 0.001);
    float width = mix(0.05, 0.16, 1.0 - detail);
    float line = 1.0 - smoothstep(width, width + aa * 2.0, fracv);
    col = mix(col, col * (1.0 - 0.32 * detail), line);

    float macro = sin(v_world_pos.x * 0.02) * sin(v_world_pos.z * 0.02);
    col *= (1.0 + 0.14 * macro * detail);

    float slope = clamp(1.0 - n.y, 0.0, 1.0);
    col *= (1.0 - 0.38 * slope * detail);

    float fog_amount = smoothstep(u_fog_start, u_fog_end, dist);
    vec3 fog_col = vec3(0.70, 0.80, 0.92);

    float cf = clamp(u_chunk_fade, 0.0, 1.0);
    col = mix(fog_col, col, cf);

    col = mix(col, fog_col, fog_amount);
    f_color = vec4(col, 1.0);
}"""

def shader_sources(ctx_version_code: int) -> tuple[str, str]:
    ver = _pick_glsl_version(ctx_version_code)
    prefix = f"#version {ver}\n"
    return prefix + _VERT_BODY, prefix + _FRAG_BODY
