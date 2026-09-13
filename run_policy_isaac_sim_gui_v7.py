"""
run_policy_isaac_sim_gui_v7.py
--------------------------------
Watch the trained PPO policy (rov_pipeline_policy_v7.pt) fly the ROV along
the real /World/Pipeline centerline inside the actual Isaac Sim GUI, on the
real rov_in_water.usda scene -- rov_physics_controller.ROVController applies
the real Fossen forces (buoyancy, damping, thrust) each PhysX step, exactly
as it would with a human/manual thrust command; this script just supplies
the thrust command every step from the trained network instead.

Same launch convention as this project's other Isaac Sim scripts
(line_tracking_controller.py's BLUEROV_GUI switch):

    source /home/mcschin1/env_isaacsim/bin/activate
    export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
    cd /home/mcschin1/Downloads/nvidia_rov/rov_pipeline_v7
    python run_policy_isaac_sim_gui_v7.py            # opens the Isaac Sim window
    HEADLESS=1 python run_policy_isaac_sim_gui_v7.py  # no window, just logs

The policy was trained at a fixed dt=0.05s (20 Hz) RK4 step (see
rov_pipeline_rl_v7.py); Isaac Sim's PhysicsScene here runs at the default
60 Hz, so the policy is queried every 3rd physics step and its thrust
command held constant in between (zero-order hold), matching the training
timestep exactly.
"""

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

USD_PATH = os.path.join(HERE, "rov_in_water.usda")
CHECKPOINT_PATH = os.path.join(HERE, "rov_pipeline_policy_v7.pt")

PHYSICS_HZ = 60.0
POLICY_DT = 0.05  # must match rov_pipeline_rl_v7.py's FossenROVDynamics dt
POLICY_EVERY = round(POLICY_DT * PHYSICS_HZ)  # = 3


def euler_zyx_from_R(R):
    """Inverse of FossenROVDynamics._rotation_matrices for a single 3x3
    body->world rotation matrix -- must match that convention exactly so the
    live observation matches what the policy was trained on."""
    theta = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    ctheta = np.cos(theta)
    if abs(ctheta) > 1e-6:
        phi = np.arctan2(R[2, 1], R[2, 2])
        psi = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock, not expected in normal pipeline-tracking flight
        phi = np.arctan2(-R[1, 2], R[1, 1])
        psi = 0.0
    return phi, theta, psi


def main():
    from isaacsim import SimulationApp

    headless = os.environ.get("HEADLESS", "0") == "1"
    simulation_app = SimulationApp({"headless": headless})

    import omni.timeline
    import omni.usd

    from rov_physics_controller import ROVController
    from rov_pipeline_rl_v7 import (
        SCENE, FossenROVDynamics, PipelineTrackingEnv, ActorCritic, DEVICE,
    )

    ctx = omni.usd.get_context()
    ctx.open_stage(USD_PATH)
    stage = ctx.get_stage()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    controller = ROVController(prim_path="/World/ROV", stage=stage)

    # Build an env instance purely to reuse its exact observation-encoding
    # math (centerline projection, nearest-rock, body-frame transforms) --
    # no stepping of its own RK4 dynamics happens here; the real physics
    # comes from PhysX + controller's Fossen force application.
    dynamics = FossenROVDynamics(SCENE)
    # rock_xy_jitter=0/rock_radius_jitter_frac=0: this observation-encoder
    # env must see the REAL rock positions from the actual open stage, not
    # a domain-randomized proxy (randomization is a training-time-only
    # technique -- see PipelineTrackingEnv's docstring/point 3).
    obs_env = PipelineTrackingEnv(SCENE, dynamics, n_envs=1, max_episode_steps=10 ** 9,
                                   rock_xy_jitter=0.0, rock_radius_jitter_frac=0.0)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"]).to(DEVICE)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    max_episode_steps = ckpt["episode_budget"]["max_episode_steps"]

    def read_live_state():
        pos, R = controller.get_world_pose()
        phi, theta, psi = euler_zyx_from_R(R)
        nu = controller._get_body_velocity()  # already body-frame, see ROVController
        state = np.zeros((1, 12))
        state[0, 0:3] = pos
        state[0, 3:6] = [phi, theta, psi]
        state[0, 6:12] = nu
        return state

    n_steps = max_episode_steps * POLICY_EVERY
    log_every = 60
    action = np.zeros((1, 6))

    # Safety metrics tracked over the WHOLE run, for the explicit
    # "did it flip / did it keep >=1m from the pipe" checks requested.
    flip_threshold_deg = np.rad2deg(obs_env.flip_threshold)
    min_dist_to_pipe = np.inf
    max_abs_roll_deg = 0.0
    max_abs_pitch_deg = 0.0
    flip_event, gap_violation_event = False, False
    outcome = "timed_out"

    print(f"Flying trained policy for up to {max_episode_steps} policy steps "
          f"({n_steps} physics steps at {PHYSICS_HZ:.0f} Hz)...")
    print(f"Safety requirements: min gap to pipe >= {obs_env.standoff_target:.2f} m, "
          f"|roll|/|pitch| < {flip_threshold_deg:.0f} deg (flip threshold)")

    for i in range(n_steps):
        if i % POLICY_EVERY == 0:
            obs_env.state = read_live_state()
            obs, s, cross_h, cross_v, dist_to_pipe, heading_alignment, rock_dist = obs_env._make_obs()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                mean_action, _, _, _ = net.act(obs_t, deterministic=True)
            action = np.clip(mean_action.cpu().numpy(), -1.0, 1.0)

            roll_deg = np.rad2deg(obs_env.state[0, 3])
            pitch_deg = np.rad2deg(obs_env.state[0, 4])
            min_dist_to_pipe = min(min_dist_to_pipe, float(dist_to_pipe[0]))
            max_abs_roll_deg = max(max_abs_roll_deg, abs(roll_deg))
            max_abs_pitch_deg = max(max_abs_pitch_deg, abs(pitch_deg))
            if abs(roll_deg) > flip_threshold_deg or abs(pitch_deg) > flip_threshold_deg:
                flip_event = True
            if dist_to_pipe[0] < obs_env.standoff_target:
                gap_violation_event = True

            if i % log_every == 0:
                pos = obs_env.state[0, 0:3]
                print(f"policy_step {i // POLICY_EVERY:4d}  t={i / PHYSICS_HZ:6.2f}s  "
                      f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f})  "
                      f"along={s[0]:6.2f}/{obs_env.total_len:5.2f}m  "
                      f"dist_to_pipe={dist_to_pipe[0]:5.2f}m  heading_align={heading_alignment[0]:5.2f}  "
                      f"roll={roll_deg:6.1f}deg  pitch={pitch_deg:6.1f}deg  "
                      f"rock_dist={rock_dist[0]:6.2f}  thrust={np.round(action[0], 2)}")

            if flip_event:
                print(f"FLIP DETECTED at t={i / PHYSICS_HZ:.2f}s "
                      f"(roll={roll_deg:.1f}deg, pitch={pitch_deg:.1f}deg) -- stopping.")
                outcome = "flipped"
                break

            if s[0] >= obs_env.total_len - obs_env.waypoint_radius:
                print(f"Reached the end of the pipeline at t={i / PHYSICS_HZ:.2f}s.")
                outcome = "success"
                break

        controller.set_thrust_command(action[0])
        simulation_app.update()

    timeline.stop()

    print("\n" + "=" * 70)
    print(f"Outcome: {outcome}")
    print(f"Min gap to pipe over the run: {min_dist_to_pipe:.3f} m "
          f"(required >= {obs_env.standoff_target:.2f} m) -- "
          f"{'OK' if not gap_violation_event else 'VIOLATED'}")
    print(f"Max |roll|: {max_abs_roll_deg:.1f} deg, max |pitch|: {max_abs_pitch_deg:.1f} deg "
          f"(flip threshold {flip_threshold_deg:.0f} deg) -- "
          f"{'OK, did not flip' if not flip_event else 'FLIPPED'}")
    print("=" * 70)

    if not headless:
        print("Window stays open (navigable) until you close it.")
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
