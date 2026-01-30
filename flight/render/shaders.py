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
in float in_water;

uniform mat4 u_proj;
uniform mat4 u_view;

out vec3 v_world_pos;
out vec3 v_norm;
out float v_height;
out float v_water;

void main() {
    v_world_pos = in_pos;
    v_norm = in_norm;
    v_height = in_pos.y;
    v_water = in_water;
    gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
}
"""

_FRAG_BODY = """in vec3 v_world_pos;
in vec3 v_norm;
in float v_height;
in float v_water;

uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;
uniform float u_fog_start;
uniform float u_fog_end;
uniform float u_chunk_fade;

uniform sampler2D u_tex_grass;
uniform sampler2D u_tex_rock;
uniform float u_tex_scale;
uniform float u_use_textures;

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

    // v0.4 (Variant B): stable texture detail.
    // - use mipmapped tile textures
    // - keep scale low to avoid high-frequency shimmer
    vec2 uv = v_world_pos.xz * u_tex_scale;
    vec3 tex_g = texture(u_tex_grass, uv).rgb;
    vec3 tex_r = texture(u_tex_rock, uv).rgb;
    float slope = 1.0 - clamp(n.y, 0.0, 1.0);
    float rock_w = clamp(smoothstep(0.18, 0.55, slope) + smoothstep(18.0, 38.0, v_height), 0.0, 1.0);
    vec3 tex_mix = mix(tex_g, tex_r, rock_w);
    base = mix(base, base * (0.75 + 0.65 * tex_mix), clamp(u_use_textures, 0.0, 1.0));

    // Water overlay (v0.3):
    //  - v_water ~1.0 => lake
    //  - v_water (0..~0.75) => river strength
    // Clean, low-frequency mask only (Variant A stability).
    float lake = smoothstep(0.90, 1.00, v_water);
    float river = smoothstep(0.10, 0.65, v_water) * (1.0 - lake);
    vec3 river_col = vec3(0.10, 0.32, 0.60);
    vec3 lake_col  = vec3(0.05, 0.18, 0.36);
    vec3 water_col = mix(river_col, lake_col, lake);
    float water_amt = clamp(river + lake, 0.0, 1.0);
    base = mix(base, water_col, water_amt);

    // Forward distance (camera moves +Z)
    float dist = max(v_world_pos.z - u_cam_pos.z, 0.0);

    // Simple lighting: more ambient for readability (Variant A)
    float ambient = 0.55;
    vec3 col = base * (ambient + 0.90 * diff);

    // Fog
    float fog_amount = smoothstep(u_fog_start, u_fog_end, dist);
    vec3 fog_col = vec3(0.70, 0.80, 0.92);

    // Product LOD: far ring gets additional "fade into fog" to hide resolution changes.
    float extra = clamp(1.0 - u_chunk_fade, 0.0, 1.0);
    fog_amount = clamp(fog_amount + extra * 0.25, 0.0, 1.0);
    col = mix(col, fog_col, fog_amount);

    f_color = vec4(col, 1.0);
}"""

def shader_sources(ctx_version_code: int) -> tuple[str, str]:
    ver = _pick_glsl_version(ctx_version_code)
    prefix = f"#version {ver}\n"
    return prefix + _VERT_BODY, prefix + _FRAG_BODY
