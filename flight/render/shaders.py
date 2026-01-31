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
uniform float u_time;
uniform float u_fog_start;
uniform float u_fog_end;

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

vec3 sky_color(vec3 dir) {
    // Approximate the sky quad: horizon -> zenith
    vec3 horizon = vec3(0.78, 0.86, 0.96);
    vec3 zenith  = vec3(0.40, 0.60, 0.85);
    float t = smoothstep(-0.05, 0.85, dir.y);
    return mix(horizon, zenith, t);
}

float wave_height(vec2 p) {
    // Low-frequency, stable water ripples (no shimmer).
    float t = u_time;
    float w = 0.0;
    w += sin(dot(p, vec2(0.010, 0.015)) * 6.283 + t * 0.9) * 0.60;
    w += sin(dot(p, vec2(-0.017, 0.008)) * 6.283 + t * 1.2) * 0.35;
    w += sin(dot(p, vec2(0.006, -0.020)) * 6.283 + t * 0.7) * 0.25;
    return w;
}

void main() {
    vec3 n = normalize(v_norm);
    vec3 l = normalize(u_light_dir);
    float diff = max(dot(n, l), 0.0);

    vec3 base = height_color(v_height);

    // Snow: by height + slope (less snow on steep cliffs)
    float slope = clamp(n.y, 0.0, 1.0);
    float snow_h = smoothstep(55.0, 85.0, v_height);
    float snow_s = smoothstep(0.55, 0.90, slope);
    float snow = snow_h * snow_s;
    vec3 snow_col = vec3(0.92, 0.94, 0.98);
    base = mix(base, snow_col, snow);

    vec3 land_base = base;

    // Water overlay (lakes + rivers)
    //  - v_water ~1.0 => lake
    //  - v_water (0..~0.75) => river strength
    float lake = smoothstep(0.90, 1.00, v_water);
    float river = smoothstep(0.10, 0.65, v_water) * (1.0 - lake);
    float water_amt = clamp(river + lake, 0.0, 1.0);

    vec3 river_col = vec3(0.10, 0.32, 0.60);
    vec3 lake_col  = vec3(0.05, 0.18, 0.36);
    vec3 water_col = mix(river_col, lake_col, lake);

    // Ripple normal (cheap finite diff)
    vec2 p = v_world_pos.xz;
    float eps = 1.25;
    float h0 = wave_height(p);
    float hx = wave_height(p + vec2(eps, 0.0));
    float hz = wave_height(p + vec2(0.0, eps));
    vec3 wn = normalize(vec3((h0 - hx) / eps, 1.0, (h0 - hz) / eps));

    // Fresnel + pseudo-reflection (sky only)
    vec3 v = normalize(u_cam_pos - v_world_pos);
    float ndv = clamp(dot(wn, v), 0.0, 1.0);
    float fres = pow(1.0 - ndv, 5.0);
    vec3 r = reflect(-v, wn);
    vec3 refl = sky_color(r);

    // A small specular highlight from the main light
    vec3 h = normalize(v + l);
    float spec = pow(max(dot(wn, h), 0.0), 64.0) * 0.65;

    vec3 water_shaded = mix(water_col, refl, clamp(fres * 0.85, 0.0, 1.0));
    water_shaded += vec3(1.0, 0.95, 0.85) * spec;

    base = mix(land_base, water_shaded, water_amt);

    // Horizontal distance in any direction (XZ plane)
    vec2 d = v_world_pos.xz - u_cam_pos.xz;
    float dist = length(d);

    // Simple lighting: more ambient for readability (Variant A)
    float ambient = 0.55;
    vec3 col = base * (ambient + 0.90 * diff);

    // Fog
    float fog_amount = smoothstep(u_fog_start, u_fog_end, dist);
    vec3 fog_col = vec3(0.70, 0.80, 0.92);
    col = mix(col, fog_col, fog_amount);

    f_color = vec4(col, 1.0);
}"""

def shader_sources(ctx_version_code: int) -> tuple[str, str]:
    ver = _pick_glsl_version(ctx_version_code)
    prefix = f"#version {ver}\n"
    return prefix + _VERT_BODY, prefix + _FRAG_BODY
