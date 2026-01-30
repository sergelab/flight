from __future__ import annotations

# Window
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS_CAP = 0  # 0 = uncapped

# Flight
DEFAULT_SEED = 12345
DEFAULT_SPEED = 20.0
DEFAULT_HEIGHT_OFFSET = 15.0
HEIGHT_SMOOTH_K = 8.0  # larger = faster follow

# Terrain / chunks
CHUNK_WORLD_SIZE = 64.0
CHUNK_RES = 64  # vertices per side
CHUNKS_X = 5  # corridor width in chunks (must be odd)
CHUNKS_Z_BEHIND = 1
CHUNKS_Z_AHEAD = 6

# Rendering
FOV_DEG = 70.0
NEAR = 0.1
FAR = 800.0
FOG_START = 150.0
FOG_END = 600.0
LIGHT_DIR = (0.35, 0.9, 0.2)  # will be normalized in shader
