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
from flight.world.world import World, WorldParams
from flight.util.math import normalize

def _init_pygame_gl() -> None:
    pygame.init()
    # macOS: request modern core profile context
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

def _project_to_ndc(proj: np.ndarray, view: np.ndarray, world_point: np.ndarray) -> tuple[float, float]:
    p = np.array([world_point[0], world_point[1], world_point[2], 1.0], dtype=np.float32)
    clip = (proj @ (view @ p))
    w = float(clip[3]) if float(clip[3]) != 0.0 else 1.0
    ndc_x = float(clip[0] / w)
    ndc_y = float(clip[1] / w)
    return ndc_x, ndc_y

def run_app(*, seed: int, speed: float, height_offset: float, wireframe: bool, debug: bool, noise_mode: str, chunk_res: int, chunk_size: float, fog_start: float, fog_end: float) -> None:
    _init_pygame_gl()

    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
    pygame.display.set_caption(f"flight v0.2 (seed={seed})")

    try:
        ctx = moderngl.create_context()
    except Exception as e:
        pygame.quit()
        raise RuntimeError(
            "Failed to create ModernGL context (need OpenGL 3.2+). "
            "Ensure you run from a normal GUI session."
        ) from e

    if debug:
        print(f"[flight] moderngl ctx version_code={ctx.version_code} vendor={ctx.info.get('GL_VENDOR')} renderer={ctx.info.get('GL_RENDERER')}")

    ctx.viewport = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    if wireframe:
        ctx.wireframe = True

    renderer = Renderer(ctx, WINDOW_WIDTH, WINDOW_HEIGHT)

    hp = HeightProvider(seed=seed, mode=str(noise_mode))
    cam = CameraRail(speed=speed, height_offset=height_offset, smooth_k=HEIGHT_SMOOTH_K)
    cam.z = -30.0  # start slightly back so forward view has terrain

    world = World(
        ctx,
        WorldParams(
            chunk_res=int(chunk_res),
            chunk_world_size=float(chunk_size),
            chunks_x=CHUNKS_X,
            chunks_z_behind=CHUNKS_Z_BEHIND,
            chunks_z_ahead=CHUNKS_Z_AHEAD,
        ),
        hp,
    )

    # Prime requests and warmup
    world.update_requests(cam.x, cam.z)
    world.warmup(renderer.prog, min_chunks=24, timeout_s=2.0)

    clock = pygame.time.Clock()
    running = True
    last_t = time.perf_counter()
    start_t = last_t
    last_log = last_t
    last_hud = last_t
    fps_est = 0.0

    light_dir = normalize(np.array(LIGHT_DIR, dtype=np.float32))

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

            # Faster uploads for first 2 seconds, then stable
            max_upload = 8 if (now - start_t) < 2.0 else 3
            world.ingest_ready(renderer.prog, max_per_frame=max_upload)

            renderer.begin_frame()
            view = cam.view_matrix()
            # Sun position on screen (project light direction into NDC)
            sun_world = cam.eye() + light_dir * 1000.0
            sun_ndc = _project_to_ndc(renderer.proj, view, sun_world)
            renderer.draw_sky(sun_ndc)
            renderer.set_common_uniforms(
                view=view,
                cam_pos=cam.eye(),
                light_dir=light_dir,
                fog_start=fog_start,
                fog_end=fog_end,
            )
            world.draw(renderer)

            if debug:
                if dt > 0:
                    fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt) if fps_est > 0 else (1.0 / dt)

                if now - last_hud >= 0.1:
                    last_hud = now
                    pending_n = len(getattr(world.cm, "pending", []))
                    lines = [
                        "flight v0.2 (debug)",
                        f"seed={seed}  speed={speed:.1f}  offset={height_offset:.1f}",
                        f"z={cam.z:.1f}  y={cam.y:.1f}  fps~{fps_est:.0f}",
                        f"chunks_gpu={len(world.chunks)}  pending={pending_n}  upload/frame={max_upload}",
                    ]
                    pad = 6
                    line_h = font.get_linesize()
                    w = max(font.size(line)[0] for line in lines) + pad * 2
                    h = line_h * len(lines) + pad * 2
                    surf = pygame.Surface((w, h), pygame.SRCALPHA)
                    surf.fill((0, 0, 0, 120))
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
                    pending_n = len(getattr(world.cm, "pending", []))
                    print(f"[flight] z={cam.z:.1f} y={cam.y:.1f} chunks_gpu={len(world.chunks)} pending={pending_n}")

            pygame.display.flip()

            if FPS_CAP and FPS_CAP > 0:
                clock.tick(FPS_CAP)
            else:
                clock.tick()
    finally:
        world.shutdown()
        renderer.release()
        pygame.quit()
