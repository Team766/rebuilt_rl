"""Game constants and configuration for FRC 2026 REBUILT ball shooter."""

import numpy as np

# Physics
GRAVITY = 9.81  # m/s^2

# Air resistance parameters
AIR_DENSITY = 1.225           # kg/m³ (at sea level, 15°C)
DRAG_COEFFICIENT = 0.47       # dimensionless (sphere)
BALL_MASS = 0.27              # kg (typical FRC cargo ball)

# Field dimensions (meters)
FIELD_LENGTH = 16.5  # 54 feet
FIELD_WIDTH = 8.2    # 27 feet

# Alliance zone (robot spawn area)
ALLIANCE_ZONE_DEPTH = 4.03   # 158.6 inches
ALLIANCE_ZONE_WIDTH = 8.07   # 317.7 inches

# HUB (target)
HUB_DISTANCE_FROM_WALL = 4.03  # 158.6 inches - distance from alliance wall to HUB center
HUB_OPENING_HEIGHT = 1.83      # 72 inches - center height of opening
HUB_OPENING_WIDTH = 1.06       # 41.7 inches - hexagonal opening width
HUB_OPENING_HALF_WIDTH = HUB_OPENING_WIDTH / 2  # 0.53m

# Ball (FUEL)
BALL_DIAMETER = 0.15   # 5.91 inches
BALL_RADIUS = BALL_DIAMETER / 2  # 0.075m

# Robot
LAUNCH_HEIGHT = 0.5    # meters - fixed launch height
MIN_DISTANCE_FROM_HUB = 0.5  # meters - minimum spawn distance from HUB

# Action space bounds
VELOCITY_MIN = 5.0     # m/s
VELOCITY_MAX = 25.0    # m/s
VELOCITY_BINS = 10     # number of discrete velocity options

ANGLE_MIN_DEG = 10.0   # degrees (elevation)
ANGLE_MAX_DEG = 80.0   # degrees (elevation)
ANGLE_BINS = 15        # number of discrete elevation angle options

# Azimuth (turret) angle settings
AZIMUTH_MIN_DEG = -90.0   # degrees (left of center)
AZIMUTH_MAX_DEG = 90.0    # degrees (right of center)
AZIMUTH_BINS = 180        # 1 degree resolution

# Derived values
VELOCITY_STEP = (VELOCITY_MAX - VELOCITY_MIN) / (VELOCITY_BINS - 1)
ANGLE_STEP_DEG = (ANGLE_MAX_DEG - ANGLE_MIN_DEG) / (ANGLE_BINS - 1)
AZIMUTH_STEP_DEG = (AZIMUTH_MAX_DEG - AZIMUTH_MIN_DEG) / (AZIMUTH_BINS - 1)

# HUB entry bounds (accounting for ball radius)
# Ball center must be within these heights for clean entry
HUB_ENTRY_MIN = HUB_OPENING_HEIGHT - HUB_OPENING_HALF_WIDTH + BALL_RADIUS  # ~1.305m
HUB_ENTRY_MAX = HUB_OPENING_HEIGHT + HUB_OPENING_HALF_WIDTH - BALL_RADIUS  # ~2.285m

# Reward shaping
REWARD_HIT_BASE = 1.0      # base reward for hitting target
REWARD_HIT_CENTER = 1.0    # bonus for center hit (total max = 2.0)
REWARD_MISS_SCALE = -0.5   # scale factor for miss penalty


def get_velocity_from_bin(bin_idx: int) -> float:
    """Convert velocity bin index to actual velocity."""
    return VELOCITY_MIN + bin_idx * VELOCITY_STEP


def get_elevation_from_bin(bin_idx: int) -> float:
    """Convert elevation angle bin index to actual angle in radians."""
    angle_deg = ANGLE_MIN_DEG + bin_idx * ANGLE_STEP_DEG
    return np.deg2rad(angle_deg)


def get_azimuth_from_bin(bin_idx: int) -> float:
    """Convert azimuth bin index to actual angle in radians."""
    angle_deg = AZIMUTH_MIN_DEG + bin_idx * AZIMUTH_STEP_DEG
    return np.deg2rad(angle_deg)


# Legacy function for backward compatibility
def get_angle_from_bin(bin_idx: int) -> float:
    """Convert angle bin index to actual angle in radians (elevation)."""
    return get_elevation_from_bin(bin_idx)


def action_to_velocity_angle(action: int) -> tuple[float, float]:
    """Convert flat action index to (velocity, elevation) tuple.

    Legacy function for 2D environment (no azimuth).
    Action space is velocity_bins × angle_bins = 150 total actions.
    action = vel_bin * ANGLE_BINS + angle_bin
    """
    vel_bin = action // ANGLE_BINS
    angle_bin = action % ANGLE_BINS
    return get_velocity_from_bin(vel_bin), get_elevation_from_bin(angle_bin)


def action_to_velocity_elevation_azimuth(action: int) -> tuple[float, float, float]:
    """Convert flat action index to (velocity, elevation, azimuth) tuple.

    Action space is velocity_bins × elevation_bins × azimuth_bins = 27,000 total actions.
    action = vel_bin * (ANGLE_BINS * AZIMUTH_BINS) + elev_bin * AZIMUTH_BINS + azim_bin
    """
    vel_bin = action // (ANGLE_BINS * AZIMUTH_BINS)
    remainder = action % (ANGLE_BINS * AZIMUTH_BINS)
    elev_bin = remainder // AZIMUTH_BINS
    azim_bin = remainder % AZIMUTH_BINS

    return (
        get_velocity_from_bin(vel_bin),
        get_elevation_from_bin(elev_bin),
        get_azimuth_from_bin(azim_bin),
    )


def velocity_elevation_azimuth_to_action(
    velocity: float, elevation_rad: float, azimuth_rad: float
) -> int:
    """Convert (velocity, elevation, azimuth) to flat action index."""
    vel_bin = int(round((velocity - VELOCITY_MIN) / VELOCITY_STEP))
    vel_bin = np.clip(vel_bin, 0, VELOCITY_BINS - 1)

    elev_deg = np.rad2deg(elevation_rad)
    elev_bin = int(round((elev_deg - ANGLE_MIN_DEG) / ANGLE_STEP_DEG))
    elev_bin = np.clip(elev_bin, 0, ANGLE_BINS - 1)

    azim_deg = np.rad2deg(azimuth_rad)
    azim_bin = int(round((azim_deg - AZIMUTH_MIN_DEG) / AZIMUTH_STEP_DEG))
    azim_bin = np.clip(azim_bin, 0, AZIMUTH_BINS - 1)

    return vel_bin * (ANGLE_BINS * AZIMUTH_BINS) + elev_bin * AZIMUTH_BINS + azim_bin


def velocity_angle_to_action(velocity: float, angle_rad: float) -> int:
    """Convert (velocity, angle) to flat action index (legacy 2D version)."""
    vel_bin = int(round((velocity - VELOCITY_MIN) / VELOCITY_STEP))
    vel_bin = np.clip(vel_bin, 0, VELOCITY_BINS - 1)

    angle_deg = np.rad2deg(angle_rad)
    angle_bin = int(round((angle_deg - ANGLE_MIN_DEG) / ANGLE_STEP_DEG))
    angle_bin = np.clip(angle_bin, 0, ANGLE_BINS - 1)

    return vel_bin * ANGLE_BINS + angle_bin


# Total action space sizes
TOTAL_ACTIONS_2D = VELOCITY_BINS * ANGLE_BINS  # 150
TOTAL_ACTIONS_3D = VELOCITY_BINS * ANGLE_BINS * AZIMUTH_BINS  # 27,000
