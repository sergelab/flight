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

    # Request a modern core profile context (important on macOS).
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

def run_app(*, seed: int, speed: float, height_offset: float, wireframe: bool) -> None:
    _init_pygame_gl()

    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
    pygame.display.set_caption(f"flight v0.1 (seed={seed})")

    # Create ModernGL context on top of pygame's current GL context
    try:
        ctx = moderngl.create_context()
    except Exception as e:
        pygame.quit()
        raise RuntimeError(
            "Failed to create ModernGL context. "
            "On macOS, ensure a Core OpenGL 3.2+ context is available. "
            "Try updating pygame/moderngl and run from a normal GUI session."
        ) from e

    ctx.viewport = (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

    if wireframe:
        ctx.wireframe = True

    renderer = Renderer(ctx, WINDOW_WIDTH, WINDOW_HEIGHT)

    hp = HeightProvider(seed=seed)
    cam = CameraRail(speed=speed, height_offset=height_offset, smooth_k=HEIGHT_SMOOTH_K)

    world = World(
        ctx,
        WorldParams(
            chunk_res=CHUNK_RES,
            chunk_world_size=CHUNK_WORLD_SIZE,
            chunks_x=CHUNKS_X,
            chunks_z_behind=CHUNKS_Z_BEHIND,
            chunks_z_ahead=CHUNKS_Z_AHEAD,
        ),
        hp.height_at,
    )

    clock = pygame.time.Clock()
    running = True
    last_t = time.perf_counter()

    light_dir = normalize(np.array(LIGHT_DIR, dtype=np.float32))

    try:
        while running:
            # dt
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            # clamp dt to avoid crazy jumps after window drag
            dt = min(dt, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    w, h = max(64, event.w), max(64, event.h)
                    screen = pygame.display.set_mode((w, h), flags)
                    renderer.resize(w, h)

            cam.update(dt, hp.height_at)

            # Update world chunk requests and ingest ready chunks (GPU upload)
            world.update_requests(cam.x, cam.z)
            world.ingest_ready(renderer.prog, max_per_frame=2)

            # Render
            renderer.begin_frame()
            view = cam.view_matrix()
            renderer.set_common_uniforms(
                view=view,
                cam_pos=cam.eye(),
                light_dir=light_dir,
                fog_start=FOG_START,
                fog_end=FOG_END,
            )
            world.draw(renderer)

            pygame.display.flip()

            if FPS_CAP and FPS_CAP > 0:
                clock.tick(FPS_CAP)
            else:
                clock.tick()  # just yields

    finally:
        world.shutdown()
        pygame.quit()
