from __future__ import annotations

import time
import numpy as np
import pygame
import moderngl

from flight.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS_CAP,
    CHUNKS_X, CHUNKS_Z_BEHIND, CHUNKS_Z_AHEAD,
    HEIGHT_SMOOTH_K, LIGHT_DIR,
    APP_VERSION,
    DEFAULT_MAX_YAW_RATE_SLOW,
    DEFAULT_MAX_YAW_RATE_FAST,
    DEFAULT_TURN_RATE,
)
from flight.render.camera import CameraRail, CameraFlight
from flight.render.renderer import Renderer
from flight.world.height import HeightProvider
from flight.world.world import WorldParams
from flight.world.lod_world import LODWorld, LODParams
from flight.util.math import normalize

def _init_pygame_gl() -> None:
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

def _surface_to_rgba_bytes(surf: pygame.Surface) -> tuple[bytes, int, int]:
    s = surf.convert_alpha()
    w, h = s.get_size()
    data = pygame.image.tostring(s, "RGBA", False)
    return data, w, h

def run_app(
    *,
    seed: int,
    max_speed: float,
    accel: float,
    brake: float,
    drag: float,
    height_offset: float,
    wireframe: bool,
    debug: bool,
    target_fps: int,
    lod: bool,
    noise_mode: str,
    chunk_res: int,
    chunk_size: float,
    fog_start: float,
    fog_end: float,
    trees: bool,
    tree_density: float,
    auto: bool,
    turn_rate: float,
    yaw_rate_slow: float,
    yaw_rate_fast: float,
    yaw_accel: float,
    yaw_decel: float,
    yaw_drag: float,
    bank_gain: float,
    bank_max: float,
    bank_smooth_k: float,
    input_smooth_k: float,
    cam_yaw_smooth_k: float,
    pitch_gain: float,
    pitch_max: float,
    pitch_smooth_k: float,
) -> None:
    _init_pygame_gl()

    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
    pygame.display.set_caption(f"flight v{APP_VERSION} (seed={seed})")

    try:
        ctx = moderngl.create_context()
    except Exception as e:
        pygame.quit()
        raise RuntimeError("Failed to create ModernGL context (need OpenGL 3.2+)") from e

    if debug:
        print(f"[flight] moderngl ctx version_code={ctx.version_code} vendor={ctx.info.get('GL_VENDOR')} renderer={ctx.info.get('GL_RENDERER')}")

    ctx.viewport = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    if wireframe:
        ctx.wireframe = True

    renderer = Renderer(ctx, WINDOW_WIDTH, WINDOW_HEIGHT)

    hp = HeightProvider(seed=seed, mode=str(noise_mode))
    # Camera
    if auto:
        cam: CameraRail | CameraFlight = CameraRail(speed=max_speed, height_offset=height_offset, smooth_k=HEIGHT_SMOOTH_K)
        cam.z = -30.0
    else:
        # Compatibility: if user explicitly tweaks --turn-rate (v0.8) but leaves yaw caps default,
        # reuse it as a single cap.
        if (
            abs(yaw_rate_slow - DEFAULT_MAX_YAW_RATE_SLOW) < 1e-6
            and abs(yaw_rate_fast - DEFAULT_MAX_YAW_RATE_FAST) < 1e-6
            and abs(turn_rate - DEFAULT_TURN_RATE) > 1e-6
        ):
            yaw_rate_slow = float(turn_rate)
            yaw_rate_fast = float(turn_rate)

        cam = CameraFlight(
            max_speed=max_speed,
            accel=accel,
            brake=brake,
            drag=drag,
            max_yaw_rate_slow=yaw_rate_slow,
            max_yaw_rate_fast=yaw_rate_fast,
            yaw_accel=yaw_accel,
            yaw_decel=yaw_decel,
            yaw_drag=yaw_drag,
            bank_gain=bank_gain,
            bank_max=bank_max,
            bank_smooth_k=bank_smooth_k,
            input_smooth_k=input_smooth_k,
            cam_yaw_smooth_k=cam_yaw_smooth_k,
            pitch_gain=pitch_gain,
            pitch_max=pitch_max,
            pitch_smooth_k=pitch_smooth_k,
            height_offset=height_offset,
            smooth_k=HEIGHT_SMOOTH_K,
        )
        cam.z = -30.0

    # LOD params (A: FPS first)
    near = WorldParams(
        chunk_res=int(chunk_res),
        chunk_world_size=float(chunk_size),
        chunks_x=int(CHUNKS_X),
        chunks_z_behind=int(CHUNKS_Z_BEHIND),
        chunks_z_ahead=int(CHUNKS_Z_AHEAD),
        skirts=True,
        skirt_depth=80.0,
        trees_enabled=bool(trees),
        tree_density=float(tree_density),
    )
    far_res = max(16, int(chunk_res // 2))
    far = WorldParams(
        chunk_res=int(far_res),
        chunk_world_size=float(chunk_size),
        chunks_x=int(CHUNKS_X * 2 + 1),
        chunks_z_behind=int(CHUNKS_Z_BEHIND * 2),
        chunks_z_ahead=int(CHUNKS_Z_AHEAD * 3),
        skirts=True,
        skirt_depth=120.0,
        trees_enabled=False,
        tree_density=0.0,
    )
    world = LODWorld(ctx, LODParams(near=near, far=far, far_update_every_n_frames=2), hp) if lod else LODWorld(ctx, LODParams(near=near, far=near, far_update_every_n_frames=999999), hp)

    # Prime + warmup
    world.update_requests(cam.x, cam.z)
    world.warmup(renderer, timeout_s=2.0)

    clock = pygame.time.Clock()
    running = True
    last_t = time.perf_counter()
    start_t = last_t
    last_log = last_t
    last_hud = last_t

    light_dir = normalize(np.array(LIGHT_DIR, dtype=np.float32))

    pygame.font.init()
    font = pygame.font.SysFont("Menlo", 16) or pygame.font.Font(None, 16)
    fps_est = 0.0

    # Adaptive streaming parameters
    target = max(15, int(target_fps))
    max_upload = 6  # initial guess

    try:
        while running:
            now = time.perf_counter()
            dt = min(now - last_t, 0.05)
            last_t = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    w, h = max(64, event.w), max(64, event.h)
                    pygame.display.set_mode((w, h), flags)
                    renderer.resize(w, h)

            # Controls
            if isinstance(cam, CameraFlight):
                keys = pygame.key.get_pressed()
                fwd = 0.0
                if keys[pygame.K_UP]:
                    fwd += 1.0
                if keys[pygame.K_DOWN]:
                    fwd -= 1.0

                # NOTE: right must turn right, left must turn left
                turn = 0.0
                if keys[pygame.K_LEFT]:
                    turn -= 1.0
                if keys[pygame.K_RIGHT]:
                    turn += 1.0

                cam.update(dt, hp.height_at, forward=fwd, turn=turn)
            else:
                cam.update(dt, hp.height_at)

            # Update requests
            world.update_requests(cam.x, cam.z)

            # Adaptive upload: if FPS below target -> reduce uploads; above -> increase slightly.
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_est = (0.9 * fps_est + 0.1 * inst_fps) if fps_est > 0 else inst_fps

            if (now - start_t) < 2.0:
                max_upload = 10
            else:
                if fps_est < target * 0.85:
                    max_upload = max(1, max_upload - 1)
                elif fps_est > target * 1.05:
                    max_upload = min(14, max_upload + 1)

            world.ingest_ready(renderer, max_per_frame=max_upload)

            # Render
            renderer.begin_frame()

            # Sky (if available)
            view = cam.view_matrix()
            if hasattr(renderer, "draw_sky") and hasattr(renderer, "proj"):
                sun_world = cam.eye() + light_dir * 1000.0
                p = np.array([sun_world[0], sun_world[1], sun_world[2], 1.0], dtype=np.float32)
                clip = (renderer.proj @ (view @ p))
                wv = float(clip[3]) if float(clip[3]) != 0.0 else 1.0
                sun_ndc = (float(clip[0] / wv), float(clip[1] / wv))
                renderer.draw_sky(sun_ndc)

            renderer.set_common_uniforms(
                view=view,
                cam_pos=cam.eye(),
                light_dir=light_dir,
                fog_start=float(fog_start),
                fog_end=float(fog_end),
                time_s=float(now - start_t),
            )
            world.draw(renderer)

            # HUD/logs
            if debug and hasattr(renderer, "hud_update_rgba") and hasattr(renderer, "draw_hud"):
                if now - last_hud >= 0.12:
                    last_hud = now
                    pending_near = len(getattr(world.near.cm, "pending", []))
                    pending_far = len(getattr(world.far.cm, "pending", []))
                    lines = [
                        f"flight v{APP_VERSION} (core flight feel)",
                        f"seed={seed} noise={noise_mode} lod={'on' if lod else 'off'} target_fps={target} auto={'on' if auto else 'off'}",
                        (
                            f"z={cam.z:.1f} y={cam.y:.1f} fps~{fps_est:.0f} upload/frame={max_upload}"
                            if not isinstance(cam, CameraFlight)
                            else f"z={cam.z:.1f} y={cam.y:.1f} v={cam.speed:.1f} yaw_rate={cam.yaw_rate:.2f} bank={cam.bank:.2f} fps~{fps_est:.0f} upload/frame={max_upload}"
                        ),
                        f"near: res={near.chunk_res} chunks={len(world.near.chunks)} pending={pending_near}",
                        f"far:  res={far.chunk_res} chunks={len(world.far.chunks)} pending={pending_far}",
                        f"fog={fog_start:.0f}->{fog_end:.0f} chunk_size={chunk_size:.1f}",
                        f"water_level={getattr(hp, 'water_level', 0.0):.1f}",
                    ]
                    pad = 6
                    line_h = font.get_linesize()
                    w = max(font.size(line)[0] for line in lines) + pad * 2
                    h = line_h * len(lines) + pad * 2
                    surf = pygame.Surface((w, h), pygame.SRCALPHA)
                    surf.fill((0, 0, 0, 130))
                    y = pad
                    for line in lines:
                        img = font.render(line, True, (255, 255, 255))
                        surf.blit(img, (pad, y))
                        y += line_h
                    rgba, tw, th = _surface_to_rgba_bytes(surf)
                    renderer.hud_update_rgba(rgba, tw, th)
                renderer.draw_hud()

                if now - last_log >= 1.0:
                    last_log = now
                    print(f"[flight] fps~{fps_est:.0f} upload={max_upload} near_chunks={len(world.near.chunks)} far_chunks={len(world.far.chunks)}")

            pygame.display.flip()

            if FPS_CAP and FPS_CAP > 0:
                clock.tick(FPS_CAP)
            else:
                clock.tick()
    finally:
        world.shutdown()
        renderer.release()
        pygame.quit()
