from __future__ import annotations

import time
import numpy as np
import pygame
import moderngl

from flight.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS_CAP,
    CHUNK_RES, CHUNK_WORLD_SIZE, CHUNKS_X, CHUNKS_Z_BEHIND, CHUNKS_Z_AHEAD,
    HEIGHT_SMOOTH_K, FOG_START, FOG_END, LIGHT_DIR,
)
from flight.render.camera import CameraRail
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
    speed: float,
    height_offset: float,
    wireframe: bool,
    debug: bool,
    target_fps: int,
    lod: bool,
    noise_mode: str,
    chunk_res: int = CHUNK_RES,
    chunk_size: float = CHUNK_WORLD_SIZE,
    fog_start: float = FOG_START,
    fog_end: float = FOG_END,
) -> None:
    _init_pygame_gl()

    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
    pygame.display.set_caption(f"flight v0.3.1 (B) seed={seed}")

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
    cam = CameraRail(speed=speed, height_offset=height_offset, smooth_k=HEIGHT_SMOOTH_K)
    cam.z = -30.0

    near = WorldParams(
        chunk_res=int(chunk_res),
        chunk_world_size=float(chunk_size),
        chunks_x=int(CHUNKS_X),
        chunks_z_behind=int(CHUNKS_Z_BEHIND),
        chunks_z_ahead=int(CHUNKS_Z_AHEAD),
    )
    far_res = max(16, int(chunk_res // 2))
    far = WorldParams(
        chunk_res=int(far_res),
        chunk_world_size=float(chunk_size),
        chunks_x=int(CHUNKS_X * 2 + 1),
        chunks_z_behind=int(CHUNKS_Z_BEHIND * 2),
        chunks_z_ahead=int(CHUNKS_Z_AHEAD * 3),
    )
    world = LODWorld(ctx, LODParams(near=near, far=far, far_update_every_n_frames=2), hp) if lod else None
    if world is None:
        # fallback: use near-only via LODWorld wrapper
        world = LODWorld(ctx, LODParams(near=near, far=near, far_update_every_n_frames=999999), hp)

    world.update_requests(cam.x, cam.z)
    world.warmup(renderer.prog, timeout_s=2.0)

    clock = pygame.time.Clock()
    running = True
    last_t = time.perf_counter()
    start_t = last_t
    last_hud = last_t
    last_log = last_t
    fps_est = 0.0

    light_dir = normalize(np.array(LIGHT_DIR, dtype=np.float32))
    target = max(15, int(target_fps))
    max_upload = 6

    pygame.font.init()
    font = pygame.font.SysFont("Menlo", 16) or pygame.font.Font(None, 16)

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

            cam.update(dt, hp.height_at)
            world.update_requests(cam.x, cam.z)

            if dt > 0:
                inst = 1.0 / dt
                fps_est = (0.9 * fps_est + 0.1 * inst) if fps_est > 0 else inst

            if (now - start_t) < 2.0:
                max_upload = 10
            else:
                if fps_est < target * 0.85:
                    max_upload = max(1, max_upload - 1)
                elif fps_est > target * 1.05:
                    max_upload = min(14, max_upload + 1)

            world.ingest_ready(renderer.prog, max_per_frame=max_upload)

            renderer.begin_frame()

            view = cam.view_matrix()
            # optional sky if renderer supports it
            if hasattr(renderer, "draw_sky") and hasattr(renderer, "proj"):
                sun_world = cam.eye() + light_dir * 1000.0
                p = np.array([sun_world[0], sun_world[1], sun_world[2], 1.0], dtype=np.float32)
                clip = (renderer.proj @ (view @ p))
                wv = float(clip[3]) if float(clip[3]) != 0.0 else 1.0
                renderer.draw_sky((float(clip[0] / wv), float(clip[1] / wv)))

            renderer.set_common_uniforms(
                view=view,
                cam_pos=cam.eye(),
                light_dir=light_dir,
                fog_start=float(fog_start),
                fog_end=float(fog_end),
            )
            world.draw(renderer, cam_z=cam.z, chunk_size=float(chunk_size))

            if debug and hasattr(renderer, "hud_update_rgba") and hasattr(renderer, "draw_hud"):
                if now - last_hud >= 0.12:
                    last_hud = now
                    pending_near = len(getattr(world.near.cm, "pending", []))
                    pending_far = len(getattr(world.far.cm, "pending", []))
                    lines = [
                        "flight v0.3.1 (B: smoother)",
                        f"lod={'on' if lod else 'off'} noise={noise_mode} target_fps={target}",
                        f"fps~{fps_est:.0f} upload/frame={max_upload} fog={fog_start:.0f}->{fog_end:.0f}",
                        f"near: res={near.chunk_res} chunks={len(world.near.chunks)} pending={pending_near}",
                        f"far:  res={far.chunk_res} chunks={len(world.far.chunks)} pending={pending_far}",
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
