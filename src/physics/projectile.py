"""2D projectile motion physics for ball shooter simulation."""

import numpy as np
from dataclasses import dataclass

from ..config import (
    GRAVITY,
    LAUNCH_HEIGHT,
    HUB_OPENING_HEIGHT,
    HUB_OPENING_HALF_WIDTH,
    HUB_ENTRY_MIN,
    HUB_ENTRY_MAX,
    BALL_RADIUS,
)


@dataclass
class TrajectoryResult:
    """Result of trajectory computation."""

    hit: bool                    # Did the ball enter the HUB?
    height_at_target: float      # Ball height when crossing target distance
    velocity_y_at_target: float  # Vertical velocity at target (negative = descending)
    miss_distance: float         # How far off vertically (0 if hit)
    center_distance: float       # Distance from center of opening (0 = perfect center)
    trajectory_x: np.ndarray     # X positions along trajectory (for visualization)
    trajectory_y: np.ndarray     # Y positions along trajectory (for visualization)


@dataclass
class TrajectoryResult3D:
    """Result of 3D trajectory computation (with azimuth)."""

    hit: bool                    # Did the ball enter the HUB?
    height_at_target: float      # Ball height when crossing target distance
    velocity_y_at_target: float  # Vertical velocity at target (negative = descending)
    lateral_offset: float        # Lateral miss distance (0 = center)
    vertical_miss: float         # Vertical miss distance (0 if within opening)
    lateral_miss: float          # Lateral miss distance (0 if within opening)
    total_miss_distance: float   # Combined miss metric
    center_distance: float       # 2D distance from center of opening
    trajectory_x: np.ndarray     # X positions along trajectory
    trajectory_y: np.ndarray     # Y positions along trajectory


def compute_trajectory(
    velocity: float,
    angle: float,
    target_distance: float,
    dt: float = 0.001,
) -> TrajectoryResult:
    """Compute 2D projectile trajectory and check if it enters the HUB.

    Args:
        velocity: Launch velocity in m/s
        angle: Launch angle in radians (from horizontal)
        target_distance: Distance to target (HUB) in meters
        dt: Time step for trajectory computation

    Returns:
        TrajectoryResult with hit status, heights, and trajectory data
    """
    # Initial conditions
    vx = velocity * np.cos(angle)
    vy = velocity * np.sin(angle)
    x = 0.0
    y = LAUNCH_HEIGHT

    # Store trajectory points
    xs = [x]
    ys = [y]
    vys = [vy]  # Track vertical velocity too

    # Track if we've crossed the target
    height_at_target = None
    vy_at_target = None

    # Simulate until ball hits ground or goes way past target
    max_x = target_distance * 2  # Don't simulate forever
    prev_x = x
    prev_y = y
    prev_vy = vy

    while y >= 0 and x <= max_x:
        # Update position
        x += vx * dt
        y += vy * dt - 0.5 * GRAVITY * dt**2

        # Update velocity (only y changes due to gravity)
        vy -= GRAVITY * dt

        # Check if we're crossing the target distance
        if prev_x < target_distance <= x and height_at_target is None:
            # Interpolate to find exact height at target
            if vx > 0:
                t_cross = (target_distance - prev_x) / vx
                height_at_target = prev_y + prev_vy * t_cross - 0.5 * GRAVITY * t_cross**2
                vy_at_target = prev_vy - GRAVITY * t_cross
            else:
                height_at_target = prev_y
                vy_at_target = prev_vy

        # Store for next iteration
        prev_x = x
        prev_y = y
        prev_vy = vy

        xs.append(x)
        ys.append(y)
        vys.append(vy)

    # Convert to arrays
    trajectory_x = np.array(xs)
    trajectory_y = np.array(ys)

    # If we never crossed the target (fell short), compute miss distance
    if height_at_target is None:
        # Ball fell short - use the last position before hitting ground
        # or extrapolate where it would have been
        max_reach = trajectory_x[-1]
        if max_reach < target_distance:
            # Ball fell short - big miss
            height_at_target = 0.0
            vy_at_target = -abs(vys[-1])  # Ensure descending
        else:
            # Shouldn't happen, but fallback
            height_at_target = trajectory_y[-1]
            vy_at_target = vys[-1]

    # Check hit conditions
    hit, miss_distance = check_hub_entry(height_at_target, vy_at_target)

    # If ball fell short, add extra penalty based on how short
    if height_at_target == 0.0 and trajectory_x[-1] < target_distance:
        shortfall = target_distance - trajectory_x[-1]
        miss_distance = max(miss_distance, shortfall + HUB_ENTRY_MIN)

    # Compute center distance
    center_distance = abs(height_at_target - HUB_OPENING_HEIGHT)

    return TrajectoryResult(
        hit=hit,
        height_at_target=height_at_target,
        velocity_y_at_target=vy_at_target,
        miss_distance=miss_distance,
        center_distance=center_distance,
        trajectory_x=trajectory_x,
        trajectory_y=trajectory_y,
    )


def check_hub_entry(height: float, vy: float) -> tuple[bool, float]:
    """Check if ball enters HUB (basketball-style, from above).

    Args:
        height: Ball center height at target distance
        vy: Vertical velocity at target (negative = descending)

    Returns:
        (hit, miss_distance) tuple
    """
    # Must be descending (negative vy)
    if vy >= 0:
        # Ball is still going up - will overshoot
        miss_distance = height - HUB_ENTRY_MAX if height > HUB_ENTRY_MAX else HUB_ENTRY_MAX - height
        return False, abs(miss_distance) + 1.0  # Extra penalty for wrong direction

    # Check if height is within valid entry range
    if HUB_ENTRY_MIN <= height <= HUB_ENTRY_MAX:
        return True, 0.0

    # Compute miss distance
    if height < HUB_ENTRY_MIN:
        miss_distance = HUB_ENTRY_MIN - height  # Too low
    else:
        miss_distance = height - HUB_ENTRY_MAX  # Too high

    return False, miss_distance


def compute_trajectory_3d(
    velocity: float,
    elevation: float,
    azimuth: float,
    target_distance: float,
    target_bearing: float,
    dt: float = 0.001,
) -> TrajectoryResult3D:
    """Compute 3D projectile trajectory with azimuth aiming.

    The vertical trajectory uses standard 2D projectile motion.
    The azimuth determines lateral offset at the target.

    Args:
        velocity: Launch velocity in m/s
        elevation: Launch elevation angle in radians (from horizontal)
        azimuth: Turret azimuth angle in radians (absolute, not relative to target)
        target_distance: Distance to target (HUB) in meters
        target_bearing: Bearing to target in radians (what azimuth should be)
        dt: Time step for trajectory computation

    Returns:
        TrajectoryResult3D with hit status and 3D miss information
    """
    # Compute 2D trajectory (elevation plane)
    result_2d = compute_trajectory(velocity, elevation, target_distance, dt)

    # Compute azimuth error and lateral offset
    azimuth_error = azimuth - target_bearing
    # Normalize to [-pi, pi]
    while azimuth_error > np.pi:
        azimuth_error -= 2 * np.pi
    while azimuth_error < -np.pi:
        azimuth_error += 2 * np.pi

    # Lateral offset at target distance
    lateral_offset = target_distance * np.sin(azimuth_error)

    # Check vertical hit (same as 2D)
    vertical_hit, vertical_miss = check_hub_entry(
        result_2d.height_at_target, result_2d.velocity_y_at_target
    )

    # Check lateral hit (within HUB radius, accounting for ball radius)
    effective_radius = HUB_OPENING_HALF_WIDTH - BALL_RADIUS
    lateral_miss = max(0.0, abs(lateral_offset) - effective_radius)
    lateral_hit = lateral_miss == 0.0

    # Overall hit requires both vertical and lateral
    hit = vertical_hit and lateral_hit

    # Compute total miss distance (Euclidean)
    if hit:
        total_miss_distance = 0.0
    else:
        total_miss_distance = np.sqrt(vertical_miss**2 + lateral_miss**2)

    # Compute center distance (2D distance from center of opening)
    vertical_center_dist = abs(result_2d.height_at_target - HUB_OPENING_HEIGHT)
    center_distance = np.sqrt(vertical_center_dist**2 + lateral_offset**2)

    return TrajectoryResult3D(
        hit=hit,
        height_at_target=result_2d.height_at_target,
        velocity_y_at_target=result_2d.velocity_y_at_target,
        lateral_offset=lateral_offset,
        vertical_miss=vertical_miss if not vertical_hit else 0.0,
        lateral_miss=lateral_miss,
        total_miss_distance=total_miss_distance,
        center_distance=center_distance,
        trajectory_x=result_2d.trajectory_x,
        trajectory_y=result_2d.trajectory_y,
    )


def compute_optimal_angle(velocity: float, target_distance: float) -> float | None:
    """Compute optimal launch angle for given velocity and distance.

    Uses the projectile motion equation to find angle that lands at HUB opening height.
    This is for reference/validation, not used in RL training.

    Args:
        velocity: Launch velocity in m/s
        target_distance: Distance to target in meters

    Returns:
        Optimal angle in radians, or None if impossible
    """
    # Target height relative to launch height
    delta_y = HUB_OPENING_HEIGHT - LAUNCH_HEIGHT

    g = GRAVITY
    d = target_distance
    v = velocity
    v2 = v * v

    # From projectile motion, height at distance d:
    # y = h₀ + d*tan(θ) - g*d²*(1 + tan²(θ)) / (2*v²)
    #
    # Let u = tan(θ), solve for y = HUB_OPENING_HEIGHT:
    # (g*d²/2v²)*u² - d*u + (delta_y + g*d²/2v²) = 0

    a = (g * d * d) / (2 * v2)
    b = -d
    c = delta_y + (g * d * d) / (2 * v2)

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return None  # Target unreachable with this velocity

    sqrt_disc = np.sqrt(discriminant)

    # Two solutions (high and low arc)
    tan_theta_1 = (-b + sqrt_disc) / (2 * a)
    tan_theta_2 = (-b - sqrt_disc) / (2 * a)

    angles = []
    for tan_theta in [tan_theta_1, tan_theta_2]:
        if tan_theta > 0:  # Valid positive angle
            angle = np.arctan(tan_theta)
            if 0 < angle < np.pi / 2:  # Between 0 and 90 degrees
                angles.append(angle)

    # Check which gives descending trajectory at target
    # Prefer low arc (more typical for FRC shooters)
    angles.sort()  # Low arc first

    for angle in angles:
        result = compute_trajectory(velocity, angle, target_distance)
        if result.velocity_y_at_target < 0:  # Descending
            return angle

    return None
