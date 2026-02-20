# Shot Simulator Guide

This guide explains how to run and use the move-and-shoot simulator in `scripts/shot_simulator.html`, including AutoShoot debugging.

## What this tool does
- Simulates 3D ball flight to the HUB with gravity and optional air drag.
- Supports robot motion during shots via robot velocity inheritance (`vx`, `vy`).
- Supports wall reflection (bounce) at alliance-zone boundaries.
- Supports two shot-command modes:
  - **Manual**: velocity/elevation/azimuth sliders.
  - **AutoShoot**: command solve ported from `ShooterUtils.java`.

## Start the simulator
1. From repo root, run:
   - `cd /home/cpadwick/code/rebuilt_rl`
   - `/home/cpadwick/code/rebuilt_rl/.venv/bin/python -m http.server 8000`
2. Open in browser:
   - `http://localhost:8000/scripts/shot_simulator.html`

## Main controls
- **Space**: fire a shot (uses AutoShoot solve when AutoShoot is enabled).
- **R**: reset robot position.
- **WASD / Arrow keys**: manual robot movement (only when Velocity Motion is disabled).

## Move-and-shoot workflow
1. Set **Robot Vx** and **Robot Vy** (range `-1.0` to `1.0`, default `0.1`).
2. Enable **Velocity Motion**.
3. Keep **Wall Reflection** enabled if you want bouncing off walls.
4. Press **Space** to fire while moving.

Notes:
- With Velocity Motion enabled, keyboard translation is disabled intentionally.
- Robot velocity is inherited into the ball trajectory model.

## AutoShoot workflow
1. Enable **AutoShoot (ShooterUtils)**.
2. Manual controls (velocity/elevation/azimuth) become disabled and grayed out.
3. Press **Space** to fire using autoshoot outputs.
4. Inspect **AutoShoot Debug** panel for solve internals.

## AutoShoot Debug fields
- `robot_vx`, `robot_vy`: robot velocity used in solve.
- `x_hub`, `y_hub`, `dist_hub`: raw robot-to-hub geometry.
- `flight_time`: estimated time of flight from initial solve.
- `target_x`, `target_y`, `target_dist`: lead-compensated target geometry.
- `launch_v`, `hood_init`, `hood_final`, `turret`: final command outputs.

## Physics/testing options
- **Air Resistance** toggle and drag/mass sliders are available in the UI.
- Built-in physics regression test mode:
  - `http://localhost:8000/scripts/shot_simulator.html?test=1`

## Troubleshooting
- **Black screen**: do a hard refresh after code changes.
- **Robot not moving**: set non-zero `vx`/`vy` and enable **Velocity Motion**.
- **AutoShoot missing**: use the debug panel values to inspect lead geometry and command outputs.
