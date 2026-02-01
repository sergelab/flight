from __future__ import annotations

# Window
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS_CAP = 0  # 0 = uncapped

# App
APP_VERSION = "0.9.3"

# Flight
DEFAULT_SEED = 12345
DEFAULT_SPEED = 20.0  # legacy name; now interpreted as max forward speed
DEFAULT_HEIGHT_OFFSET = 15.0
HEIGHT_SMOOTH_K = 8.0  # larger = faster follow
DEFAULT_MIN_CLEARANCE = 0.8  # minimal clearance above ground/water to avoid clipping

# Controls
# v0.8: manual flight controls (arrow keys)
DEFAULT_AUTO = False
DEFAULT_TURN_RATE = 1.8  # rad/sec (legacy: v0.8 direct yaw)

# v0.9: "body" flight feel (inertial speed + inertial yaw)
# Linear
DEFAULT_MAX_SPEED = DEFAULT_SPEED
DEFAULT_ACCEL = 4.5
DEFAULT_BRAKE = 6.5
DEFAULT_DRAG = 0.18

# Angular (yaw)
DEFAULT_MAX_YAW_RATE_SLOW = 1.6
DEFAULT_MAX_YAW_RATE_FAST = 0.65
DEFAULT_YAW_ACCEL = 2.2
DEFAULT_YAW_DECEL = 3.0
DEFAULT_YAW_DRAG = 1.6

# Camera reactions
DEFAULT_BANK_GAIN = 1.35
DEFAULT_BANK_MAX = 0.6  # radians (~34 deg)
DEFAULT_BANK_SMOOTH_K = 5.5

# Input & view smoothing (critical for "weight" with digital arrows)
# k is the exponential smoothing constant; smaller = heavier/laggier response.
DEFAULT_INPUT_SMOOTH_K = 3.5
DEFAULT_CAM_YAW_SMOOTH_K = 3.0

# Vertical (height) control
DEFAULT_CLIMB_RATE = 10.0  # units/sec for q/a lift control

# Longitudinal acceleration cue (visual pitch). Values are subtle.
DEFAULT_PITCH_GAIN = 0.012
DEFAULT_PITCH_MAX = 0.22  # radians (~12.6 deg)
DEFAULT_PITCH_SMOOTH_K = 4.0

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
FOG_END = 220.0
LIGHT_DIR = (0.35, 0.9, 0.2)  # will be normalized in shader

# v0.2: tunables (can be overridden via CLI)
DEFAULT_CHUNK_RES = CHUNK_RES
DEFAULT_CHUNK_WORLD_SIZE = CHUNK_WORLD_SIZE
DEFAULT_FOG_START = FOG_START
DEFAULT_FOG_END = FOG_END

# v0.3 defaults
DEFAULT_TARGET_FPS = 60
DEFAULT_LOD = False  # v0.3 Variant A: stability baseline
DEFAULT_NOISE = "fast"

# v0.7 defaults: forests (variant C)
DEFAULT_TREES = True
DEFAULT_TREE_DENSITY = 1.0
