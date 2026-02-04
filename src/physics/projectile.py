"""Projectile motion physics for ball shooter simulation (2D and 3D)."""

from dataclasses import dataclass

import numpy as np

from ..config import (
    AIR_DENSITY,
    BALL_MASS,
    BALL_RADIUS,
    DRAG_COEFFICIENT,
    GRAVITY,
    HUB_ENTRY_MAX,
    HUB_ENTRY_MIN,
    HUB_OPENING_HALF_WIDTH,
    HUB_OPENING_HEIGHT,
    LAUNCH_HEIGHT,
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
    air_resistance: bool = False,
) -> TrajectoryResult:
    """Compute 2D projectile trajectory and check if it enters the HUB.

    Args:
        velocity: Launch velocity in m/s
        angle: Launch angle in radians (from horizontal)
        target_distance: Distance to target (HUB) in meters
        dt: Time step for trajectory computation
        air_resistance: Whether to include air resistance (drag)

    Returns:
        TrajectoryResult with hit status, heights, and trajectory data
    """
    # Initial conditions
    vx = velocity * np.cos(angle)
    vy = velocity * np.sin(angle)
    x = 0.0
    y = LAUNCH_HEIGHT

    # Air resistance coefficient: k = 0.5 * rho * Cd * A / m
    # where A = pi * r^2 (cross-sectional area)
    if air_resistance:
        cross_section = np.pi * BALL_RADIUS**2
        drag_coeff = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * cross_section / BALL_MASS

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
        if air_resistance:
            # Compute velocity magnitude
            v_mag = np.sqrt(vx**2 + vy**2)

            # Drag acceleration (opposes velocity direction)
            # a_drag = -k * v * v_hat = -k * v^2 * (v/|v|) = -k * |v| * v
            if v_mag > 0:
                ax_drag = -drag_coeff * v_mag * vx
                ay_drag = -drag_coeff * v_mag * vy
            else:
                ax_drag = 0.0
                ay_drag = 0.0

            # Update velocities with drag and gravity
            vx += ax_drag * dt
            vy += (ay_drag - GRAVITY) * dt

            # Update position
            x += vx * dt
            y += vy * dt
        else:
            # Simple projectile motion (no air resistance)
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
                if air_resistance:
                    # With drag, use linear interpolation (small dt approximation)
                    height_at_target = prev_y + prev_vy * t_cross
                    vy_at_target = prev_vy + (vy - prev_vy) * (t_cross / dt) if dt > 0 else vy
                else:
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
    air_resistance: bool = False,
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
        air_resistance: Whether to include air resistance (drag)

    Returns:
        TrajectoryResult3D with hit status and 3D miss information
    """
    # Compute 2D trajectory (elevation plane)
    result_2d = compute_trajectory(velocity, elevation, target_distance, dt, air_resistance)

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


@dataclass
class TrajectoryResult3DMoving:
    """Result of 3D trajectory computation with robot velocity."""

    hit: bool
    height_at_target: float
    velocity_z_at_target: float
    lateral_offset: float
    vertical_miss: float
    lateral_miss: float
    total_miss_distance: float
    center_distance: float
    trajectory_x: np.ndarray  # field X coordinates
    trajectory_y: np.ndarray  # field Y coordinates (lateral)
    trajectory_z: np.ndarray  # height


def compute_trajectory_3d_moving(
    launch_velocity: float,
    elevation: float,
    azimuth: float,
    target_distance: float,
    target_bearing: float,
    robot_vx: float = 0.0,
    robot_vy: float = 0.0,
    dt: float = 0.001,
    air_resistance: bool = False,
) -> TrajectoryResult3DMoving:
    """Compute 3D trajectory where the ball inherits robot horizontal velocity.

    The ball's initial velocity is the vector sum of the launch velocity
    (decomposed by elevation and azimuth) and the robot's ground-plane velocity.

    Coordinate frame (robot at origin):
      - Bearing axis points from robot toward HUB
      - Cross-bearing axis is perpendicular (right-hand rule)
      - Z is height (vertical, up positive)

    Args:
        launch_velocity: Muzzle velocity in m/s (relative to robot)
        elevation: Launch elevation angle in radians (from horizontal)
        azimuth: Turret azimuth angle in radians (absolute field frame)
        target_distance: Distance from robot to hub center in meters
        target_bearing: Bearing from robot to hub in radians
        robot_vx: Robot velocity in field X direction (m/s)
        robot_vy: Robot velocity in field Y direction (m/s)
        dt: Time step for Euler integration
        air_resistance: Whether to include quadratic drag

    Returns:
        TrajectoryResult3DMoving with full 3D trajectory data
    """
    # Decompose launch velocity into field-frame 3D components
    v_horiz = launch_velocity * np.cos(elevation)
    vz = launch_velocity * np.sin(elevation)

    # Horizontal launch direction determined by azimuth (field frame)
    vx = v_horiz * np.cos(azimuth) + robot_vx
    vy = v_horiz * np.sin(azimuth) + robot_vy

    # Ball starts at robot position (origin), height = LAUNCH_HEIGHT
    x = 0.0
    y = 0.0
    z = LAUNCH_HEIGHT

    # Air resistance coefficient
    if air_resistance:
        cross_section = np.pi * BALL_RADIUS**2
        drag_k = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * cross_section / BALL_MASS

    xs, ys, zs = [x], [y], [z]

    # Bearing unit vectors for projection onto target plane
    cos_b = np.cos(target_bearing)
    sin_b = np.sin(target_bearing)

    # Track when ball crosses the target plane (perpendicular to bearing at target_distance)
    height_at_target = None
    vz_at_target = None
    lateral_at_target = None

    prev_d_along = 0.0
    prev_x, prev_y, prev_z, prev_vz = x, y, z, vz

    max_time = 5.0  # seconds upper bound
    t = 0.0

    while t < max_time and z >= 0:
        if air_resistance:
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            if v_mag > 0:
                ax = -drag_k * v_mag * vx
                ay = -drag_k * v_mag * vy
                az = -drag_k * v_mag * vz - GRAVITY
            else:
                ax, ay, az = 0.0, 0.0, -GRAVITY
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
        else:
            vz -= GRAVITY * dt

        x += vx * dt
        y += vy * dt
        z += vz * dt
        t += dt

        # Project position onto bearing axis
        d_along = x * cos_b + y * sin_b

        # Detect crossing of target plane
        if prev_d_along < target_distance <= d_along and height_at_target is None:
            # Linear interpolation to find exact crossing point
            if d_along > prev_d_along:
                frac = (target_distance - prev_d_along) / (d_along - prev_d_along)
            else:
                frac = 0.0
            height_at_target = prev_z + frac * (z - prev_z)
            vz_at_target = prev_vz + frac * (vz - prev_vz)
            # Lateral offset: perpendicular component to bearing
            cross_x = prev_x + frac * (x - prev_x)
            cross_y = prev_y + frac * (y - prev_y)
            lateral_at_target = -cross_x * sin_b + cross_y * cos_b

        prev_d_along = d_along
        prev_x, prev_y, prev_z, prev_vz = x, y, z, vz

        xs.append(x)
        ys.append(y)
        zs.append(z)

    # Ball never reached target plane
    if height_at_target is None:
        height_at_target = 0.0
        vz_at_target = -abs(vz) if vz != 0 else -1.0
        lateral_at_target = 0.0

    # Evaluate hit/miss using same criteria as existing code
    vertical_hit, vertical_miss = check_hub_entry(height_at_target, vz_at_target)

    effective_radius = HUB_OPENING_HALF_WIDTH - BALL_RADIUS
    lateral_miss_dist = max(0.0, abs(lateral_at_target) - effective_radius)
    lateral_hit = lateral_miss_dist == 0.0

    hit = vertical_hit and lateral_hit

    if hit:
        total_miss_distance = 0.0
    else:
        total_miss_distance = np.sqrt(vertical_miss**2 + lateral_miss_dist**2)

    vertical_center_dist = abs(height_at_target - HUB_OPENING_HEIGHT)
    center_distance = np.sqrt(vertical_center_dist**2 + lateral_at_target**2)

    return TrajectoryResult3DMoving(
        hit=hit,
        height_at_target=height_at_target,
        velocity_z_at_target=vz_at_target,
        lateral_offset=lateral_at_target,
        vertical_miss=vertical_miss if not vertical_hit else 0.0,
        lateral_miss=lateral_miss_dist,
        total_miss_distance=total_miss_distance,
        center_distance=center_distance,
        trajectory_x=np.array(xs),
        trajectory_y=np.array(ys),
        trajectory_z=np.array(zs),
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
