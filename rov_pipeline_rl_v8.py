"""
rov_pipeline_rl_v8.py
----------------------
PPO training + evaluation for autonomous pipeline tracking, using the same
6-DOF Fossen marine-craft dynamics that rov_physics_controller.py layers on
top of Isaac Sim's PhysX rigid body (ROV_Dynamics_Model_Datasheet.pdf /
rov_in_water.usda's rov:* attributes), and the same /World/Pipeline
centerline (rov:pipelinePoints), rock obstacles (rov:obstacle /
rov:obstacleRadius) and seafloor as the scene.

Why a standalone dynamics engine instead of driving live Isaac Sim:
rov_physics_controller.py's own docstring says to treat a standalone RK4
integration of the Fossen model as "the source of truth", with the PhysX
force-layering version only for the interactive visual scene -- and PPO
needs millions of fast steps, which a live PhysX/Isaac Sim session cannot
give you. So this script parses rov_in_water.usda once (pipeline points,
rock positions/radii, seafloor height, ROV mass/damping/allocation
attributes) and re-implements the exact governing equation from the
datasheet:

    M * dnu/dt = tau_thrust + D(nu) - g(eta)      (body frame)
    deta/dt    = [R(phi,theta,psi) @ nu[0:3], T(phi,theta) @ nu[3:6]]

integrated with fixed-step RK4 at dt=0.05s (datasheet Sec. 1/7), so a
trained policy's thrust commands are physically consistent with what
rov_physics_controller.py would produce inside Isaac Sim for the same
scene, and can be dropped into ROVController.set_thrust_command(...) there
for a final visual check.

Usage:
    python rov_pipeline_rl_v8.py --train                 # train PPO from scratch
    python rov_pipeline_rl_v8.py --evaluate               # run 1 full eval episode + plots
    python rov_pipeline_rl_v8.py --train --evaluate        # both, back to back
"""

import argparse
import csv
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


HERE = os.path.dirname(os.path.abspath(__file__))
USDA_PATH = os.path.join(HERE, "rov_in_water.usda")
CHECKPOINT_PATH = os.path.join(HERE, "rov_pipeline_policy_v8.pt")
LOG_DIR = os.path.join(HERE, "logs")
EVAL_DIR = os.path.join(HERE, "evaluation")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOF_ORDER = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


# ============================================================================
# 1. Parse the scene straight out of rov_in_water.usda -- single source of
#    truth for pipeline centerline, rocks, seafloor, and the ROV's Fossen
#    parameters / thruster allocation matrix, so this script can never drift
#    out of sync with the scene it is meant to fly.
# ============================================================================

def _float_list(text):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", text)]


def parse_usda_scene(path):
    src = open(path, "r").read()

    def attr(name, cast=float, count=None):
        m = re.search(rf"rov:{name}\s*=\s*([^\n]+)", src)
        if m is None:
            raise ValueError(f"attribute rov:{name} not found in {path}")
        vals = _float_list(m.group(1))
        if count is not None and len(vals) != count:
            raise ValueError(f"rov:{name} expected {count} values, got {len(vals)}")
        return vals if cast is float else [cast(v) for v in vals]

    rov = {
        "mass": attr("weight", count=1)[0] / 9.81,
        "weight": attr("weight", count=1)[0],
        "buoyancy": attr("buoyancy", count=1)[0],
        "added_mass": np.array(attr("addedMass", count=6)),
        "linear_damping": np.array(attr("linearDamping", count=6)),
        "quadratic_damping": np.array(attr("quadraticDamping", count=6)),
        "thruster_max_force": attr("thrusterMaxForce", count=1)[0],
        "allocation": np.array([
            attr("allocation:u", count=6),
            attr("allocation:v", count=6),
            attr("allocation:w", count=6),
            attr("allocation:p", count=6),
            attr("allocation:q", count=6),
            attr("allocation:r", count=6),
        ]),  # rows = DOF, cols = thrusters (6x6)
    }

    # inertia + CoM straight from the PhysicsMassAPI attrs (not rov:* prefixed)
    m = re.search(r"physics:diagonalInertia\s*=\s*\(([^)]+)\)", src)
    rov["inertia_diag"] = np.array(_float_list(m.group(1)))
    m = re.search(r"physics:centerOfMass\s*=\s*\(([^)]+)\)", src)
    rov["r_g"] = np.array(_float_list(m.group(1)))

    # ROV spawn pose: the xformOp:translate on the /World/ROV Xform prim
    # itself (the block that also carries physxRigidBody:disableGravity).
    rov_block = src[src.index('def Xform "ROV"'):]
    m = re.search(r"double3 xformOp:translate = \(([^)]+)\)", rov_block)
    rov["spawn_pos"] = np.array(_float_list(m.group(1)))

    # pipeline centerline
    m = re.search(r"rov:pipelinePoints\s*=\s*(\[[^\]]+\])", src)
    pts = _float_list(m.group(1))
    pipeline_points = np.array(pts).reshape(-1, 3)

    # rocks: each Sphere "Rock_N" block carries its own translate,
    # rov:obstacleRadius, inside the enclosing Xform's own translate (0 here,
    # but computed generally in case that ever changes).
    rocks_block_start = src.index('def Xform "Rocks"')
    rocks_block = src[rocks_block_start:src.index("def Sphere \"WP_00\"")]
    rock_positions, rock_radii = [], []
    for rock_src in re.split(r'def Sphere "Rock_\d+"', rocks_block)[1:]:
        end = re.search(r"\n\s*\}", rock_src).start()
        block = rock_src[:end]
        pos = _float_list(re.search(r"double3 xformOp:translate = \(([^)]+)\)", block).group(1))
        radius = _float_list(re.search(r"rov:obstacleRadius\s*=\s*([-\d.]+)", block).group(1))[0]
        rock_positions.append(pos)
        rock_radii.append(radius)
    rocks = {"positions": np.array(rock_positions), "radii": np.array(rock_radii)}

    # seafloor: a Cube of USD "size=1" (i.e. +-0.5 local) scaled then
    # translated -- top surface sits at translate.z + 0.5*scale.z
    m = re.search(r'def Cube "Seafloor".*?xformOp:scale = \(([^)]+)\).*?xformOp:translate = \(([^)]+)\)', src, re.S)
    scale = _float_list(m.group(1))
    translate = _float_list(m.group(2))
    seafloor_top_z = translate[2] + 0.5 * scale[2]

    return {
        "rov": rov,
        "pipeline_points": pipeline_points,
        "rocks": rocks,
        "seafloor_top_z": seafloor_top_z,
    }


SCENE = parse_usda_scene(USDA_PATH)


# ============================================================================
# 2. Fossen 6-DOF dynamics: mass matrix (from the datasheet's explicit M,
#    built here from rigid-body inertia + the parsed added-mass diagonal so
#    it reproduces the datasheet's M/M^-1 table, including the CoG-offset
#    surge<->pitch / sway<->roll coupling) + RK4 integrator, vectorized over
#    a batch of N independent environments (pure numpy -- physics is cheap,
#    PPO needs many parallel rollouts, not a single fast one).
# ============================================================================

class FossenROVDynamics:
    def __init__(self, scene, dt=0.05):
        rov = scene["rov"]
        self.dt = dt
        self.mass = rov["mass"]
        self.weight = rov["weight"]
        self.buoyancy = rov["buoyancy"]
        self.net_wb = self.weight - self.buoyancy  # W - B; negative => positively buoyant
        self.lin_damp = rov["linear_damping"]
        self.quad_damp = rov["quadratic_damping"]
        self.t_max = rov["thruster_max_force"]
        self.A = rov["allocation"]  # (6,6) rows=DOF cols=thrusters

        Ix, Iy, Iz = rov["inertia_diag"]
        rgx, rgy, rgz = rov["r_g"]
        m = self.mass

        # Rigid-body mass matrix with CoG offset (Fossen 3.44), plus the
        # diagonal added-mass terms from the datasheet -- reproduces exactly
        # the M / M^-1 tables in ROV_Dynamics_Model_Datasheet.pdf Sec. 4.
        M_rb = np.array([
            [m,      0,      0,      0,      m*rgz,  -m*rgy],
            [0,      m,      0,     -m*rgz,  0,       m*rgx],
            [0,      0,      m,      m*rgy, -m*rgx,   0],
            [0,     -m*rgz,  m*rgy,  Ix,     0,       0],
            [m*rgz,  0,     -m*rgx,  0,      Iy,      0],
            [-m*rgy, m*rgx,  0,      0,      0,       Iz],
        ])
        M_added = -np.diag(rov["added_mass"])  # datasheet lists added mass as negative coefficients
        self.M = M_rb + M_added
        self.M_inv = np.linalg.inv(self.M)

    def restoring_force(self, phi, theta):
        """g(eta) translational rows, +z-up convention (datasheet Sec. 5)."""
        net = self.net_wb  # W - B
        gx = -net * np.sin(theta)
        gy = net * np.cos(theta) * np.sin(phi)
        gz = net * np.cos(theta) * np.cos(phi)
        return gx, gy, gz

    def thruster_wrench(self, action):
        """action: (N,6) normalized thruster commands in [-1,1] -> (N,6) body wrench."""
        forces = np.clip(action, -1.0, 1.0) * self.t_max  # (N,6) per-thruster N
        return forces @ self.A.T  # (N,6): tau[n] = A @ forces[n]

    @staticmethod
    def _rotation_matrices(phi, theta, psi):
        """Body->world rotation R (N,3,3), ZYX Euler convention."""
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        N = phi.shape[0]
        R = np.zeros((N, 3, 3))
        R[:, 0, 0] = cpsi * cth
        R[:, 0, 1] = cpsi * sth * sphi - spsi * cphi
        R[:, 0, 2] = cpsi * sth * cphi + spsi * sphi
        R[:, 1, 0] = spsi * cth
        R[:, 1, 1] = spsi * sth * sphi + cpsi * cphi
        R[:, 1, 2] = spsi * sth * cphi - cpsi * sphi
        R[:, 2, 0] = -sth
        R[:, 2, 1] = cth * sphi
        R[:, 2, 2] = cth * cphi
        return R

    @staticmethod
    def _euler_rate_transform(phi, theta):
        """Euler-rate transform T (N,3,3), with theta clamped away from the
        +-90deg gimbal singularity (never reached in practice for a
        pipeline-tracking ROV, but guards RL exploration noise)."""
        theta = np.clip(theta, -1.48, 1.48)  # ~ +-85 deg
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        tth = sth / cth
        N = phi.shape[0]
        T = np.zeros((N, 3, 3))
        T[:, 0, 0] = 1.0
        T[:, 0, 1] = sphi * tth
        T[:, 0, 2] = cphi * tth
        T[:, 1, 1] = cphi
        T[:, 1, 2] = -sphi
        T[:, 2, 1] = sphi / cth
        T[:, 2, 2] = cphi / cth
        return T

    def derivative(self, state, action):
        """state: (N,12) = [pos(3), euler(3), nu(6)] -> dstate/dt (N,12)."""
        euler = state[:, 3:6]
        nu = state[:, 6:12]
        phi, theta, psi = euler[:, 0], euler[:, 1], euler[:, 2]

        damping = self.lin_damp * nu + self.quad_damp * nu * np.abs(nu)  # (N,6)
        tau_thrust = self.thruster_wrench(action)  # (N,6)
        gx, gy, gz = self.restoring_force(phi, theta)
        g_vec = np.zeros_like(nu)
        g_vec[:, 0], g_vec[:, 1], g_vec[:, 2] = gx, gy, gz

        body_force = tau_thrust + damping - g_vec  # (N,6)
        nu_dot = body_force @ self.M_inv.T  # (N,6)

        R = self._rotation_matrices(phi, theta, psi)
        T = self._euler_rate_transform(phi, theta)
        pos_dot = np.einsum("nij,nj->ni", R, nu[:, 0:3])
        euler_dot = np.einsum("nij,nj->ni", T, nu[:, 3:6])

        dstate = np.concatenate([pos_dot, euler_dot, nu_dot], axis=1)
        return dstate

    def step(self, state, action):
        """Fixed-step RK4, datasheet Sec. 1/7 (dt=0.05s)."""
        dt = self.dt
        k1 = self.derivative(state, action)
        k2 = self.derivative(state + 0.5 * dt * k1, action)
        k3 = self.derivative(state + 0.5 * dt * k2, action)
        k4 = self.derivative(state + dt * k3, action)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================================
# 3. Realistic cruise speed -> episode length, so a full traversal is always
#    possible within one episode instead of being cut short.
# ============================================================================

def compute_cruise_speed_and_episode_budget(dynamics, scene, cruise_thrust_fraction=0.6,
                                             safety_factor=8.0):
    """Cruise speed: steady-state surge speed where thrust (at a sustainable
    fraction of max, leaving margin for cross-track/attitude correction and
    obstacle avoidance) balances linear+quadratic drag. Solves
        |Xu|*u + |Xuu|*u^2 = F_cruise
    for u (the quadratic dominates at speed, matching D(nu)nu in Sec. 3 of
    the datasheet), the same balance rov_physics_controller.py's damping
    term reaches at steady state.
    """
    # Max pure-surge thrust: drive all 4 horizontal thrusters (T1..T4) at
    # full command with signs matching the allocation matrix's surge row so
    # sway/yaw net to zero -- this is the actual achievable max surge force.
    A = dynamics.A
    surge_row = A[0]
    cmd = np.sign(surge_row)
    cmd[surge_row == 0] = 0.0
    f_thrusters = cmd * dynamics.t_max
    tau = A @ f_thrusters
    f_surge_max = tau[0]

    f_cruise = cruise_thrust_fraction * f_surge_max
    Xu = abs(dynamics.lin_damp[0])
    Xuu = abs(dynamics.quad_damp[0])
    # Xuu*u^2 + Xu*u - f_cruise = 0
    cruise_speed = (-Xu + np.sqrt(Xu ** 2 + 4 * Xuu * f_cruise)) / (2 * Xuu)

    pts = scene["pipeline_points"]
    pipeline_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    spawn_to_start = float(np.linalg.norm(scene["rov"]["spawn_pos"] - pts[0]))
    total_distance = pipeline_length + spawn_to_start

    nominal_time_s = total_distance / cruise_speed
    episode_duration_s = safety_factor * nominal_time_s
    max_episode_steps = int(np.ceil(episode_duration_s / dynamics.dt))

    info = dict(
        f_surge_max=float(f_surge_max),
        f_cruise=float(f_cruise),
        cruise_speed_mps=float(cruise_speed),
        pipeline_length_m=pipeline_length,
        spawn_to_start_m=spawn_to_start,
        total_distance_m=total_distance,
        nominal_transit_time_s=float(nominal_time_s),
        safety_factor=safety_factor,
        episode_duration_s=float(episode_duration_s),
        max_episode_steps=max_episode_steps,
    )
    return info


# ============================================================================
# 4. Vectorized pipeline-tracking environment (N parallel episodes).
# ============================================================================

class PipelineTrackingEnv:
    # +2 vs v5: dist_to_pipe (3D standoff distance) and heading_alignment
    # (cosine similarity between the ROV's forward axis and the local
    # pipeline tangent), see point 1/2 of the v6 changes below.
    OBS_DIM = 25
    ACT_DIM = 6

    def __init__(self, scene, dynamics, n_envs, max_episode_steps,
                 rov_collision_radius=0.20, seafloor_margin=0.05,
                 waypoint_radius=0.30, out_of_bounds_radius=15.0,
                 k_progress=8.0, collision_penalty=50.0,
                 completion_bonus=100.0, k_energy=1.0,
                 # v6/v7: standoff tracking (point 1 + explicit "keep >=1m
                 # from the pipe" requirement) instead of hugging the
                 # centerline to zero distance. v6 used a wide dead-band
                 # ([1.5m, 3.0m] with zero gradient in between) that let the
                 # vehicle settle anywhere in it -- it drifted out to ~2m
                 # RMSE with no incentive to come closer. v7 replaces that
                 # with a continuous quadratic attractor pulling toward
                 # standoff_track_target (1.2m: a safety margin above the
                 # 1.0m hard floor, not sitting right on top of it), so the
                 # vehicle is actively pulled to a tight, consistent gap
                 # instead of merely avoiding the two extremes.
                 standoff_target=1.0, standoff_track_target=1.5, k_track=8.0,
                 success_max_dist_to_pipe=2.5,
                 # v6 point 2: heading alignment with the local pipeline
                 # tangent (camera/sonar-forward inspection posture).
                 k_heading=1.5,
                 # v6: explicit flip prevention + general attitude-stability
                 # shaping. flip_threshold_deg is the hard "this counts as
                 # a flip" cutoff on |roll| or |pitch|.
                 flip_threshold_deg=75.0, k_attitude=0.3,
                 # v6 point 3: domain-randomize rock layout every reset so
                 # the policy learns genuine avoidance instead of
                 # memorizing one fixed obstacle field.
                 rock_xy_jitter=0.3, rock_radius_jitter_frac=0.2,
                 seed=0):
        self.scene = scene
        self.dyn = dynamics
        self.n = n_envs
        self.max_steps = max_episode_steps
        self.rov_r = rov_collision_radius
        self.seafloor_limit = scene["seafloor_top_z"] + seafloor_margin
        self.waypoint_radius = waypoint_radius
        self.oob_radius = out_of_bounds_radius
        self.k_progress = k_progress
        self.collision_penalty = collision_penalty
        self.completion_bonus = completion_bonus
        self.k_energy = k_energy
        self.standoff_target = standoff_target
        self.standoff_track_target = standoff_track_target
        self.k_track = k_track
        self.success_max_dist_to_pipe = success_max_dist_to_pipe
        self.k_heading = k_heading
        self.flip_threshold = np.deg2rad(flip_threshold_deg)
        self.k_attitude = k_attitude
        self.rock_xy_jitter = rock_xy_jitter
        self.rock_radius_jitter_frac = rock_radius_jitter_frac
        self.rng = np.random.default_rng(seed)

        self.pts = scene["pipeline_points"].astype(np.float64)  # (P,3)
        seg_vec = np.diff(self.pts, axis=0)  # (P-1,3)
        seg_len = np.linalg.norm(seg_vec, axis=1)  # (P-1,)
        self.seg_vec = seg_vec
        self.seg_len = seg_len
        self.seg_dir = seg_vec / seg_len[:, None]
        self.cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.total_len = float(self.cum_len[-1])
        self.final_point = self.pts[-1]

        # Per-env rock layout (domain randomization target): base positions
        # tiled across envs, then jittered independently per env in reset().
        base_rock_pos = scene["rocks"]["positions"].astype(np.float64)  # (R,3)
        base_rock_r = scene["rocks"]["radii"].astype(np.float64)  # (R,)
        self.base_rock_pos = base_rock_pos
        self.base_rock_r = base_rock_r
        self.rock_pos = np.tile(base_rock_pos[None, :, :], (n_envs, 1, 1))  # (N,R,3)
        self.rock_r = np.tile(base_rock_r[None, :], (n_envs, 1))  # (N,R)

        self.spawn_pos = scene["rov"]["spawn_pos"].astype(np.float64)

        self.state = np.zeros((self.n, 12))
        self.prev_s = np.zeros(self.n)
        self.steps = np.zeros(self.n, dtype=int)
        self.done = np.zeros(self.n, dtype=bool)
        self.reset(np.ones(self.n, dtype=bool))

    # ---- centerline projection (along-track s, cross-track/vertical error,
    # and local tangent direction for the heading-alignment reward) --------
    def _project_to_centerline(self, pos):
        """pos: (N,3) -> s(N,), cross_h(N,), cross_v(N,), tangent(N,3)."""
        N = pos.shape[0]
        n_seg = self.seg_len.shape[0]
        best_dist2 = np.full(N, np.inf)
        best_s = np.zeros(N)
        best_perp = np.zeros((N, 3))
        best_tangent = np.zeros((N, 3))
        for j in range(n_seg):
            p0 = self.pts[j]
            d = self.seg_dir[j]
            L = self.seg_len[j]
            rel = pos - p0  # (N,3)
            t = np.clip(rel @ d, 0.0, L)  # (N,)
            proj = p0 + t[:, None] * d
            perp = pos - proj
            dist2 = np.einsum("ij,ij->i", perp, perp)
            better = dist2 < best_dist2
            best_dist2 = np.where(better, dist2, best_dist2)
            best_s = np.where(better, self.cum_len[j] + t, best_s)
            best_perp = np.where(better[:, None], perp, best_perp)
            best_tangent = np.where(better[:, None], d[None, :], best_tangent)
        cross_v = best_perp[:, 2]
        cross_h = np.sqrt(np.maximum(best_dist2 - cross_v ** 2, 0.0))
        return best_s, cross_h, cross_v, best_tangent

    def _nearest_rock(self, pos):
        """pos: (N,3) -> dist_to_surface(N,), rel_vec_world(N,3) (to center).
        Rocks are per-env (self.rock_pos/self.rock_r are (N,R,..)) so each
        parallel environment can have its own randomized obstacle layout."""
        N = pos.shape[0]
        best_dist = np.full(N, np.inf)
        best_rel = np.zeros((N, 3))
        for k in range(self.rock_pos.shape[1]):
            rel = self.rock_pos[:, k, :] - pos  # (N,3), vector FROM rov TO rock
            dist_center = np.linalg.norm(rel, axis=1)
            dist_surface = dist_center - self.rock_r[:, k]
            better = dist_surface < best_dist
            best_dist = np.where(better, dist_surface, best_dist)
            best_rel = np.where(better[:, None], rel, best_rel)
        return best_dist, best_rel

    def _body_frame_vec(self, euler, vec_world):
        R = self.dyn._rotation_matrices(euler[:, 0], euler[:, 1], euler[:, 2])
        return np.einsum("nji,nj->ni", R, vec_world)  # R^T @ vec (world->body)

    def _make_obs(self):
        pos = self.state[:, 0:3]
        euler = self.state[:, 3:6]
        nu = self.state[:, 6:12]

        s, cross_h, cross_v, tangent = self._project_to_centerline(pos)
        dist_to_goal = self.total_len - s
        dist_to_pipe = np.sqrt(cross_h ** 2 + cross_v ** 2)

        rock_dist, rock_rel_world = self._nearest_rock(pos)
        rock_rel_body = self._body_frame_vec(euler, rock_rel_world)
        rock_bearing = np.arctan2(rock_rel_body[:, 1], rock_rel_body[:, 0])

        R = self.dyn._rotation_matrices(euler[:, 0], euler[:, 1], euler[:, 2])
        forward_world = R[:, :, 0]  # body +x axis expressed in world frame
        heading_alignment = np.einsum("ij,ij->i", forward_world, tangent)

        sinphi, cosphi = np.sin(euler[:, 0]), np.cos(euler[:, 0])
        sinth, costh = np.sin(euler[:, 1]), np.cos(euler[:, 1])
        sinpsi, cospsi = np.sin(euler[:, 2]), np.cos(euler[:, 2])

        obs = np.concatenate([
            pos,                                            # 3
            np.stack([sinphi, cosphi, sinth, costh, sinpsi, cospsi], axis=1),  # 6
            nu,                                              # 6
            s[:, None], dist_to_goal[:, None],               # 2
            cross_h[:, None], cross_v[:, None],              # 2
            dist_to_pipe[:, None],                            # 1
            heading_alignment[:, None],                       # 1
            rock_dist[:, None],                              # 1
            np.stack([np.sin(rock_bearing), np.cos(rock_bearing)], axis=1),  # 2
            rock_rel_body[:, 2:3],                           # 1  (elevation component)
        ], axis=1)
        assert obs.shape[1] == self.OBS_DIM
        return obs, s, cross_h, cross_v, dist_to_pipe, heading_alignment, rock_dist

    def reset(self, mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        n = len(idx)
        self.state[idx, 0:3] = self.spawn_pos + self.rng.normal(0, 0.05, size=(n, 3))
        self.state[idx, 3] = self.rng.normal(0, np.deg2rad(3), size=n)   # roll
        self.state[idx, 4] = self.rng.normal(0, np.deg2rad(3), size=n)   # pitch
        self.state[idx, 5] = self.rng.uniform(-np.pi, np.pi, size=n)     # yaw
        self.state[idx, 6:12] = 0.0
        self.steps[idx] = 0
        self.done[idx] = False

        # v6 domain randomization: jitter each rock's xy position and
        # radius independently for these envs' new episode.
        R = self.base_rock_pos.shape[0]
        xy_jitter = self.rng.uniform(-self.rock_xy_jitter, self.rock_xy_jitter, size=(n, R, 2))
        self.rock_pos[idx, :, 0:2] = self.base_rock_pos[None, :, 0:2] + xy_jitter
        self.rock_pos[idx, :, 2] = self.base_rock_pos[None, :, 2]
        radius_scale = self.rng.uniform(1.0 - self.rock_radius_jitter_frac,
                                         1.0 + self.rock_radius_jitter_frac, size=(n, R))
        self.rock_r[idx] = self.base_rock_r[None, :] * radius_scale

        s0, _, _, _ = self._project_to_centerline(self.state[idx, 0:3])
        self.prev_s[idx] = s0

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.state = self.dyn.step(self.state, action)
        self.steps += 1

        obs, s, cross_h, cross_v, dist_to_pipe, heading_alignment, rock_dist = self._make_obs()
        pos = self.state[:, 0:3]
        roll, pitch = self.state[:, 3], self.state[:, 4]

        progress = s - self.prev_s
        self.prev_s = s

        reward = self.k_progress * progress

        # v7: continuous quadratic attractor toward standoff_track_target
        # (1.2m -- a safety margin above the 1.0m hard floor) instead of
        # v6's wide [1.5m, 3.0m] dead-band, which had zero gradient in
        # between and let the vehicle settle anywhere in it (observed RMSE
        # ~2m in v6). This actively pulls it to a tight, consistent gap.
        track_error = dist_to_pipe - self.standoff_track_target
        reward = reward - self.k_track * track_error ** 2
        pipe_violation = dist_to_pipe < self.standoff_target

        # v6 point 2: reward facing the direction of travel (camera/sonar
        # forward), heading_alignment in [-1,1] (cosine similarity).
        reward = reward + self.k_heading * heading_alignment

        # v6 flip prevention: continuous attitude-stability shaping plus a
        # hard failure if roll/pitch actually crosses the flip threshold.
        reward = reward - self.k_attitude * (roll ** 2 + pitch ** 2)
        flipped = (np.abs(roll) > self.flip_threshold) | (np.abs(pitch) > self.flip_threshold)

        # Energy/aggressiveness penalty: discourages saturating every
        # thruster at +-1 ("bang-bang" control) that worked in this fast
        # proxy sim but pinned the vehicle against real PhysX collision
        # geometry in Isaac Sim with no learned recovery. Penalizing mean
        # squared thrust pushes the policy toward smaller, smoother commands
        # wherever full thrust isn't actually needed for progress.
        reward = reward - self.k_energy * np.mean(action ** 2, axis=1)

        rock_collision = rock_dist < self.rov_r
        seafloor_collision = pos[:, 2] < self.seafloor_limit
        collided = rock_collision | seafloor_collision | pipe_violation | flipped

        # v6: success is reaching the end of the route WHILE holding the
        # tracking corridor -- not literal 3D proximity to the final point,
        # since the vehicle now deliberately keeps a >=1m standoff from the
        # pipe (including its endpoint) rather than hugging the centerline.
        reached_goal = (s >= self.total_len - self.waypoint_radius) & (dist_to_pipe <= self.success_max_dist_to_pipe)

        out_of_bounds = (np.linalg.norm(pos[:, 0:2], axis=1) > self.oob_radius) | (pos[:, 2] > -0.1)

        reward = reward - self.collision_penalty * (collided | out_of_bounds).astype(float)
        reward = reward + self.completion_bonus * reached_goal.astype(float)

        timed_out = self.steps >= self.max_steps
        done = collided | reached_goal | out_of_bounds | timed_out

        info = dict(
            success=reached_goal.copy(),
            collided=collided.copy(),
            rock_collision=rock_collision.copy(),
            seafloor_collision=seafloor_collision.copy(),
            pipe_violation=pipe_violation.copy(),
            flipped=flipped.copy(),
            out_of_bounds=out_of_bounds.copy(),
            timed_out=(timed_out & ~(collided | reached_goal | out_of_bounds)).copy(),
            along_track=s.copy(),
            cross_track=np.sqrt(cross_h ** 2 + cross_v ** 2),
            dist_to_pipe=dist_to_pipe.copy(),
            heading_alignment=heading_alignment.copy(),
            roll=roll.copy(),
            pitch=pitch.copy(),
        )

        self.done = done
        return obs, reward, done, info


# ============================================================================
# 5. PPO (actor-critic MLP, GAE, clipped surrogate) -- plain PyTorch, no
#    external RL library, so the whole pipeline is self-contained.
# ============================================================================

def mlp(sizes, activation=nn.Tanh, out_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.3  # std in ~[0.05, 1.35] -- bounded so the
# policy can't drift into the unbounded-entropy "bang-bang" exploration
# regime that saturates every thruster at +-1 and pins the vehicle against
# real collision geometry with no learned recovery.


class ActorCritic(nn.Module):
    """Tanh-squashed Gaussian policy: the network's raw (pre-tanh) output is
    what PPO's log-probs/ratios are computed on, and tanh(raw) is the actual
    [-1,1] thruster command applied to the environment -- bounded smoothly
    (unlike hard-clipping a plain Gaussian) so gradients don't vanish right
    at saturation, which is what let commands pile up at +-1 before."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.pi_mean = mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(-1.0 * torch.ones(act_dim))
        self.v = mlp([obs_dim, *hidden, 1])

    def forward(self, obs):
        mean = self.pi_mean(obs)
        std = torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
        value = self.v(obs).squeeze(-1)
        return mean, std, value

    @staticmethod
    def _tanh_log_prob(dist, raw_action, eps=1e-6):
        logp = dist.log_prob(raw_action).sum(-1)
        correction = torch.log(1 - torch.tanh(raw_action).pow(2) + eps).sum(-1)
        return logp - correction

    def act(self, obs, deterministic=False):
        """Returns (squashed_action, raw_action, logp, value). raw_action is
        what must be stored for PPO's later log-prob recomputation;
        squashed_action is what actually drives the environment."""
        mean, std, value = self.forward(obs)
        if deterministic:
            return torch.tanh(mean), mean, None, value
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        logp = self._tanh_log_prob(dist, raw_action)
        return action, raw_action, logp, value

    def evaluate_actions(self, obs, raw_action):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        logp = self._tanh_log_prob(dist, raw_action)
        entropy = dist.entropy().sum(-1)  # underlying Gaussian's entropy (standard approximation)
        return logp, entropy, value


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    lastgaelam = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nextvalues = last_values if t == T - 1 else values[t + 1]
        nextnonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def train(total_timesteps=6_000_000, n_envs=64, rollout_len=256,
          lr_start=3e-4, lr_end=3e-5, clip_ratio=0.2, train_epochs=10, minibatch_size=4096,
          vf_coef=0.5, ent_coef_start=0.01, ent_coef_end=0.001, max_grad_norm=0.5,
          gamma=0.99, lam=0.95,
          # v8: curriculum on the standoff-tracking target -- start loose
          # (1.5m, close to v6's easy dead-band edge, for stable early
          # learning) and tighten toward 1.05m (just above the 1.0m hard
          # floor) by the end of training, instead of v7's fixed 1.2m
          # target from step 1 (which made training noisier -- entropy and
          # success both oscillated hard for most of the run).
          standoff_curriculum_start=1.5, standoff_curriculum_end=1.05,
          log_every=1, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dynamics = FossenROVDynamics(SCENE)
    budget = compute_cruise_speed_and_episode_budget(dynamics, SCENE)
    print("=== Episode length sizing ===")
    for k, v in budget.items():
        print(f"  {k}: {v}")
    max_episode_steps = budget["max_episode_steps"]

    env = PipelineTrackingEnv(SCENE, dynamics, n_envs, max_episode_steps, seed=seed,
                               standoff_track_target=standoff_curriculum_start)

    net = ActorCritic(env.OBS_DIM, env.ACT_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr_start)

    obs, _, _, _, _, _, _ = env._make_obs()
    ep_return = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, dtype=int)
    ep_track_err_sum = np.zeros(n_envs)  # accumulates |dist_to_pipe - target| per step
    finished_returns, finished_lens, finished_success = [], [], []
    finished_flipped, finished_pipe_violation = [], []
    finished_track_error = []  # per-episode mean standoff-tracking error, for best-checkpoint scoring

    n_updates = int(np.ceil(total_timesteps / (n_envs * rollout_len)))
    csv_path = os.path.join(LOG_DIR, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["update", "timesteps", "mean_ep_return", "mean_ep_len",
                          "success_rate", "flip_rate", "pipe_gap_violation_rate",
                          "policy_loss", "value_loss", "entropy", "elapsed_s",
                          "standoff_track_target", "ent_coef", "lr", "mean_track_error"])

    t_start = time.time()
    global_step = 0
    # Best-checkpoint selection: PPO with a continuing entropy bonus can
    # drift AWAY from a good deterministic policy late in training (observed
    # here -- success climbed to ~100% mid-run, then entropy crept back up
    # and the final update's policy had regressed). So track the best
    # rolling (success, safety) score seen so far and save THAT checkpoint,
    # rather than trusting the very last update to be the best one.
    best_score = -np.inf
    best_update = None

    def save_checkpoint():
        torch.save({
            "model_state_dict": net.state_dict(),
            "obs_dim": env.OBS_DIM,
            "act_dim": env.ACT_DIM,
            "episode_budget": budget,
            "total_timesteps": global_step,
        }, CHECKPOINT_PATH)

    for update in range(1, n_updates + 1):
        # v8 curriculum/annealing: linearly interpolate the standoff target,
        # entropy coefficient, and learning rate across training. Tightening
        # the target only gradually (rather than fixing it at 1.2m from
        # step 1, as v7 did) keeps early learning as stable as v6's, while
        # still ending tighter than v7. Annealing entropy/lr toward small-
        # but-nonzero end values directly targets the late-training
        # oscillation seen in both v6 and v7 (entropy climbing back up after
        # the policy had already converged).
        progress = (update - 1) / max(n_updates - 1, 1)
        env.standoff_track_target = (
            standoff_curriculum_start + (standoff_curriculum_end - standoff_curriculum_start) * progress
        )
        ent_coef = ent_coef_start + (ent_coef_end - ent_coef_start) * progress
        current_lr = lr_start + (lr_end - lr_start) * progress
        for g in opt.param_groups:
            g["lr"] = current_lr

        obs_buf = np.zeros((rollout_len, n_envs, env.OBS_DIM), dtype=np.float32)
        act_buf = np.zeros((rollout_len, n_envs, env.ACT_DIM), dtype=np.float32)
        logp_buf = np.zeros((rollout_len, n_envs), dtype=np.float32)
        rew_buf = np.zeros((rollout_len, n_envs), dtype=np.float32)
        done_buf = np.zeros((rollout_len, n_envs), dtype=np.float32)
        val_buf = np.zeros((rollout_len, n_envs), dtype=np.float32)

        for t in range(rollout_len):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                action, raw_action, logp, value = net.act(obs_t)
            action_np = action.cpu().numpy()  # squashed, in [-1,1] -- drives the env

            next_obs, reward, done, info = env.step(action_np)

            obs_buf[t] = obs
            act_buf[t] = raw_action.cpu().numpy()  # pre-tanh, for PPO's log-prob recompute
            logp_buf[t] = logp.cpu().numpy()
            rew_buf[t] = reward
            done_buf[t] = done.astype(np.float32)
            val_buf[t] = value.cpu().numpy()

            ep_return += reward
            ep_len += 1
            ep_track_err_sum += info["dist_to_pipe"]
            if done.any():
                idx = np.where(done)[0]
                for i in idx:
                    finished_returns.append(ep_return[i])
                    finished_lens.append(ep_len[i])
                    finished_success.append(bool(info["success"][i]))
                    finished_flipped.append(bool(info["flipped"][i]))
                    finished_pipe_violation.append(bool(info["pipe_violation"][i]))
                    finished_track_error.append(ep_track_err_sum[i] / max(ep_len[i], 1))
                ep_return[idx] = 0.0
                ep_len[idx] = 0
                ep_track_err_sum[idx] = 0.0
                env.reset(done)
                next_obs, _, _, _, _, _, _ = env._make_obs()

            obs = next_obs
            global_step += n_envs

        with torch.no_grad():
            last_val = net.forward(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE))[2].cpu().numpy()

        adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_val, gamma=gamma, lam=lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        b_obs = torch.as_tensor(obs_buf.reshape(-1, env.OBS_DIM), device=DEVICE)
        b_act = torch.as_tensor(act_buf.reshape(-1, env.ACT_DIM), device=DEVICE)
        b_logp = torch.as_tensor(logp_buf.reshape(-1), device=DEVICE)
        b_adv = torch.as_tensor(adv.reshape(-1), device=DEVICE)
        b_ret = torch.as_tensor(ret.reshape(-1), device=DEVICE)

        n_samples = b_obs.shape[0]
        last_pi_loss = last_v_loss = last_ent = 0.0
        for epoch in range(train_epochs):
            perm = torch.randperm(n_samples, device=DEVICE)
            for start in range(0, n_samples, minibatch_size):
                mb = perm[start:start + minibatch_size]
                logp, entropy, value = net.evaluate_actions(b_obs[mb], b_act[mb])
                ratio = torch.exp(logp - b_logp[mb])
                surr1 = ratio * b_adv[mb]
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * b_adv[mb]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((value - b_ret[mb]) ** 2).mean()
                ent_loss = -entropy.mean()
                loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                opt.step()

                last_pi_loss, last_v_loss, last_ent = pi_loss.item(), v_loss.item(), -ent_loss.item()

        if update % log_every == 0:
            mean_ret = float(np.mean(finished_returns[-200:])) if finished_returns else float("nan")
            mean_len = float(np.mean(finished_lens[-200:])) if finished_lens else float("nan")
            succ = float(np.mean(finished_success[-200:])) if finished_success else float("nan")
            flip_rate = float(np.mean(finished_flipped[-200:])) if finished_flipped else float("nan")
            gap_viol_rate = float(np.mean(finished_pipe_violation[-200:])) if finished_pipe_violation else float("nan")
            mean_track_err = float(np.mean(finished_track_error[-200:])) if finished_track_error else float("nan")
            elapsed = time.time() - t_start
            print(f"update {update}/{n_updates} | steps {global_step} | "
                  f"ep_return {mean_ret:.2f} | ep_len {mean_len:.1f} | success {succ:.2f} | "
                  f"flip {flip_rate:.2f} | gap_violation {gap_viol_rate:.2f} | track_err {mean_track_err:.3f} | "
                  f"pi_loss {last_pi_loss:.4f} | v_loss {last_v_loss:.4f} | ent {last_ent:.4f} | "
                  f"target {env.standoff_track_target:.3f} | ent_coef {ent_coef:.4f} | lr {current_lr:.1e} | "
                  f"elapsed {elapsed:.1f}s")
            csv_writer.writerow([update, global_step, mean_ret, mean_len, succ, flip_rate, gap_viol_rate,
                                  last_pi_loss, last_v_loss, last_ent, elapsed,
                                  env.standoff_track_target, ent_coef, current_lr, mean_track_err])
            csv_file.flush()

            # Require a reasonable sample of episodes before trusting the
            # rolling stats; penalize unsafe checkpoints even if success
            # looks good so a flip-prone/gap-violating policy is never
            # preferred over a safer, slightly-less-successful one. Also
            # reward tighter standoff tracking (small weight, mostly a
            # tie-breaker among near-perfect candidates) -- without this,
            # v8's first run picked update 33 (score 1.0, but from very
            # early training when the curriculum target was still ~1.47m,
            # i.e. LOOSER than v7's fixed 1.2m) purely because it was the
            # first checkpoint to reach perfect success/flip/gap, entirely
            # ignoring how tight the tracking actually was.
            if len(finished_success) >= 50:
                score = succ - 0.5 * flip_rate - 0.5 * gap_viol_rate - 0.15 * mean_track_err
                if score > best_score:
                    best_score = score
                    best_update = update
                    save_checkpoint()

    csv_file.close()

    if best_update is None:
        save_checkpoint()  # fallback: nothing ever cleared the 50-episode bar
        print(f"Saved checkpoint (final update, no best-checkpoint threshold reached) to {CHECKPOINT_PATH}")
    else:
        print(f"Saved BEST checkpoint (update {best_update}/{n_updates}, score {best_score:.3f}) to {CHECKPOINT_PATH}")

    plot_training_curves(csv_path)
    return net, dynamics, max_episode_steps


def plot_training_curves(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    steps = [int(r["timesteps"]) for r in rows]
    returns = [float(r["mean_ep_return"]) for r in rows]
    lens = [float(r["mean_ep_len"]) for r in rows]
    succ = [float(r["success_rate"]) for r in rows]
    flip = [float(r["flip_rate"]) for r in rows]
    gap_viol = [float(r["pipe_gap_violation_rate"]) for r in rows]
    target = [float(r["standoff_track_target"]) for r in rows]
    ent_coef = [float(r["ent_coef"]) for r in rows]
    track_err = [float(r["mean_track_error"]) for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    axes[0].plot(steps, returns)
    axes[0].set_ylabel("mean episode return\n(last 200 episodes)")
    axes[0].set_title("PPO training curves -- ROV pipeline tracking (v8)")
    axes[1].plot(steps, lens, color="tab:orange")
    axes[1].set_ylabel("mean episode length (steps)")
    axes[2].plot(steps, succ, color="tab:green", label="success rate")
    axes[2].plot(steps, flip, color="tab:red", label="flip rate")
    axes[2].plot(steps, gap_viol, color="tab:purple", label="<1m pipe-gap violation rate")
    axes[2].set_ylabel("rate (last 200 episodes)")
    axes[2].legend(loc="center right", fontsize=8)
    axes[3].plot(steps, target, color="tab:brown", label="standoff track target (m)")
    axes[3].plot(steps, track_err, color="tab:blue", label="actual mean tracking error (m)")
    ax2 = axes[3].twinx()
    ax2.plot(steps, ent_coef, color="tab:cyan", linestyle="--", label="entropy coef")
    axes[3].set_ylabel("standoff distance (m)")
    ax2.set_ylabel("entropy coef")
    axes[3].set_xlabel("environment timesteps")
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(LOG_DIR, "training_curves.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


# ============================================================================
# 6. Evaluation: one full deterministic episode, full trajectory logged +
#    plotted end to end (not truncated).
# ============================================================================

def _outcome_from_info(info):
    if info["success"][0]:
        return "success"
    if info["flipped"][0]:
        return "flipped"
    if info["pipe_violation"][0]:
        return "pipe_gap_violation"
    if info["rock_collision"][0]:
        return "rock_collision"
    if info["seafloor_collision"][0]:
        return "seafloor_collision"
    if info["out_of_bounds"][0]:
        return "out_of_bounds"
    return "timed_out"


def evaluate(checkpoint_path=CHECKPOINT_PATH, seed=123, n_eval_episodes=20):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    dynamics = FossenROVDynamics(SCENE)
    max_episode_steps = ckpt["episode_budget"]["max_episode_steps"]

    net = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"]).to(DEVICE)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # ---- 1) one full, fully-logged episode -- the canonical inspectable
    # end-to-end trajectory deliverable (plots + CSV of every step). Rocks
    # are NOT randomized for this canonical run so it matches the fixed
    # scene shown in the Isaac Sim GUI 1:1.
    env = PipelineTrackingEnv(SCENE, dynamics, n_envs=1, max_episode_steps=max_episode_steps,
                               seed=seed, rock_xy_jitter=0.0, rock_radius_jitter_frac=0.0)
    obs, _, _, _, _, _, _ = env._make_obs()

    traj = dict(t=[], pos=[], euler=[], nu=[], action=[], reward=[],
                along_track=[], cross_track=[], dist_to_pipe=[], heading_alignment=[],
                roll_deg=[], pitch_deg=[])

    total_reward, outcome = 0.0, None
    for step in range(max_episode_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            action, _, _, _ = net.act(obs_t, deterministic=True)
        action_np = np.clip(action.cpu().numpy(), -1.0, 1.0)

        pos = env.state[0, 0:3].copy()
        euler = env.state[0, 3:6].copy()
        nu = env.state[0, 6:12].copy()

        next_obs, reward, done, info = env.step(action_np)

        traj["t"].append(step * dynamics.dt)
        traj["pos"].append(pos)
        traj["euler"].append(euler)
        traj["nu"].append(nu)
        traj["action"].append(action_np[0].copy())
        traj["reward"].append(float(reward[0]))
        traj["along_track"].append(float(info["along_track"][0]))
        traj["cross_track"].append(float(info["cross_track"][0]))
        traj["dist_to_pipe"].append(float(info["dist_to_pipe"][0]))
        traj["heading_alignment"].append(float(info["heading_alignment"][0]))
        traj["roll_deg"].append(float(np.rad2deg(info["roll"][0])))
        traj["pitch_deg"].append(float(np.rad2deg(info["pitch"][0])))
        total_reward += float(reward[0])

        obs = next_obs
        if done[0]:
            outcome = _outcome_from_info(info)
            break
    else:
        outcome = "timed_out"

    for k in traj:
        traj[k] = np.array(traj[k])

    print(f"Evaluation episode finished: outcome={outcome}, steps={len(traj['t'])}, "
          f"total_reward={total_reward:.2f}, final along-track={traj['along_track'][-1]:.2f}"
          f"/{env.total_len:.2f} m")

    save_evaluation_outputs(env, dynamics, traj, outcome, total_reward)

    # ---- 2) many quick episodes (domain-randomized rocks + spawn jitter)
    # for a real success/flip/gap-violation rate with a confidence interval,
    # instead of judging the policy off one anecdotal run.
    evaluate_multi(net, dynamics, max_episode_steps, n_episodes=n_eval_episodes, base_seed=seed + 1)

    return traj, outcome, total_reward


def _wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    half_width = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - half_width) / denom, (centre + half_width) / denom)


def evaluate_multi(net, dynamics, max_episode_steps, n_episodes=20, base_seed=1000):
    """Runs n_episodes deterministic-policy episodes, each with its own
    domain-randomized rock layout and spawn jitter (rock_xy_jitter left at
    its default), and reports success/flip/gap-violation rates with 95%
    Wilson confidence intervals -- point 4 of the v6 requirements: judge the
    policy on more than one anecdotal run."""
    results = []
    for ep in range(n_episodes):
        env = PipelineTrackingEnv(SCENE, dynamics, n_envs=1, max_episode_steps=max_episode_steps,
                                   seed=base_seed + ep)
        obs, _, _, _, _, _, _ = env._make_obs()
        min_dist_to_pipe = np.inf
        max_abs_roll = max_abs_pitch = 0.0
        outcome = "timed_out"
        for step in range(max_episode_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                action, _, _, _ = net.act(obs_t, deterministic=True)
            action_np = np.clip(action.cpu().numpy(), -1.0, 1.0)
            obs, reward, done, info = env.step(action_np)
            min_dist_to_pipe = min(min_dist_to_pipe, float(info["dist_to_pipe"][0]))
            max_abs_roll = max(max_abs_roll, abs(float(np.rad2deg(info["roll"][0]))))
            max_abs_pitch = max(max_abs_pitch, abs(float(np.rad2deg(info["pitch"][0]))))
            if done[0]:
                outcome = _outcome_from_info(info)
                break
        results.append(dict(outcome=outcome, min_dist_to_pipe=min_dist_to_pipe,
                             max_abs_roll_deg=max_abs_roll, max_abs_pitch_deg=max_abs_pitch))

    n = len(results)
    n_success = sum(r["outcome"] == "success" for r in results)
    n_flipped = sum(r["outcome"] == "flipped" for r in results)
    n_gap_violation = sum(r["outcome"] == "pipe_gap_violation" for r in results)
    success_ci = _wilson_ci(n_success, n)
    flip_ci = _wilson_ci(n_flipped, n)
    gap_ci = _wilson_ci(n_gap_violation, n)

    summary = dict(
        n_episodes=n,
        success_rate=n_success / n, success_rate_95ci=list(success_ci),
        flip_rate=n_flipped / n, flip_rate_95ci=list(flip_ci),
        pipe_gap_violation_rate=n_gap_violation / n, pipe_gap_violation_rate_95ci=list(gap_ci),
        outcomes=[r["outcome"] for r in results],
        min_dist_to_pipe_over_all_episodes=min(r["min_dist_to_pipe"] for r in results),
        max_abs_roll_deg_over_all_episodes=max(r["max_abs_roll_deg"] for r in results),
        max_abs_pitch_deg_over_all_episodes=max(r["max_abs_pitch_deg"] for r in results),
    )
    out_path = os.path.join(EVAL_DIR, "eval_multi_episode_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Multi-episode eval ({n} episodes, domain-randomized rocks): "
          f"success {summary['success_rate']:.0%} (95% CI {success_ci[0]:.0%}-{success_ci[1]:.0%}), "
          f"flip {summary['flip_rate']:.0%}, gap_violation {summary['pipe_gap_violation_rate']:.0%}")
    print(f"Saved multi-episode evaluation summary to {out_path}")
    return summary


def save_evaluation_outputs(env, dynamics, traj, outcome, total_reward):
    # full step-by-step trajectory as CSV -- the complete run, not a clip
    csv_path = os.path.join(EVAL_DIR, "eval_trajectory.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x", "y", "z", "roll", "pitch", "yaw",
                    "u", "v", "w", "p", "q", "r",
                    "T1", "T2", "T3", "T4", "T5", "T6",
                    "reward", "along_track_m", "cross_track_m",
                    "dist_to_pipe_m", "heading_alignment", "roll_deg", "pitch_deg"])
        for i in range(len(traj["t"])):
            w.writerow([
                traj["t"][i],
                *traj["pos"][i], *traj["euler"][i], *traj["nu"][i], *traj["action"][i],
                traj["reward"][i], traj["along_track"][i], traj["cross_track"][i],
                traj["dist_to_pipe"][i], traj["heading_alignment"][i],
                traj["roll_deg"][i], traj["pitch_deg"][i],
            ])
    print(f"Saved full evaluation trajectory to {csv_path}")

    max_abs_roll = float(np.max(np.abs(traj["roll_deg"]))) if len(traj["roll_deg"]) else 0.0
    max_abs_pitch = float(np.max(np.abs(traj["pitch_deg"]))) if len(traj["pitch_deg"]) else 0.0
    min_dist_to_pipe = float(np.min(traj["dist_to_pipe"])) if len(traj["dist_to_pipe"]) else None
    flip_threshold_deg = float(np.rad2deg(env.flip_threshold))

    summary = dict(
        outcome=outcome,
        total_reward=total_reward,
        n_steps=int(len(traj["t"])),
        duration_s=float(traj["t"][-1] + dynamics.dt) if len(traj["t"]) else 0.0,
        final_along_track_m=float(traj["along_track"][-1]) if len(traj["along_track"]) else 0.0,
        pipeline_length_m=env.total_len,
        mean_cross_track_m=float(np.mean(traj["cross_track"])) if len(traj["cross_track"]) else None,
        max_cross_track_m=float(np.max(traj["cross_track"])) if len(traj["cross_track"]) else None,
        mean_heading_alignment=float(np.mean(traj["heading_alignment"])) if len(traj["heading_alignment"]) else None,
        # explicit checks requested: no-flip and >=1m standoff from the pipe
        min_dist_to_pipe_m=min_dist_to_pipe,
        required_min_gap_m=env.standoff_target,
        gap_requirement_met=(min_dist_to_pipe is not None and min_dist_to_pipe >= env.standoff_target),
        max_abs_roll_deg=max_abs_roll,
        max_abs_pitch_deg=max_abs_pitch,
        flip_threshold_deg=flip_threshold_deg,
        did_not_flip=(max_abs_roll < flip_threshold_deg and max_abs_pitch < flip_threshold_deg),
    )
    with open(os.path.join(EVAL_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {os.path.join(EVAL_DIR, 'eval_summary.json')}")
    print(f"  gap requirement (>= {env.standoff_target:.1f} m): min observed "
          f"{min_dist_to_pipe:.3f} m -- {'MET' if summary['gap_requirement_met'] else 'VIOLATED'}")
    print(f"  flip check (< {flip_threshold_deg:.0f} deg): max |roll|={max_abs_roll:.1f}, "
          f"max |pitch|={max_abs_pitch:.1f} -- {'OK' if summary['did_not_flip'] else 'FLIPPED'}")

    # 3D trajectory plot: pipeline centerline, rocks, seafloor plane, full path
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    pts = env.pts
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "k-", linewidth=3, label="Pipeline centerline")

    for k in range(env.rock_pos.shape[1]):
        ax.scatter(*env.rock_pos[0, k], s=env.rock_r[0, k] * 1200, c="saddlebrown", alpha=0.6)
    ax.scatter([], [], [], c="saddlebrown", label="Rocks")

    xx, yy = np.meshgrid(np.linspace(-4, 4, 2), np.linspace(-2.5, 2.5, 2))
    zz = np.full_like(xx, env.scene["seafloor_top_z"])
    ax.plot_surface(xx, yy, zz, alpha=0.15, color="tab:brown")

    p = traj["pos"]
    ax.plot(p[:, 0], p[:, 1], p[:, 2], "b-", linewidth=1.5, label="ROV trajectory (full episode)")
    ax.scatter(*p[0], c="green", s=60, label="Start")
    ax.scatter(*p[-1], c="red" if outcome != "success" else "lime", s=80, marker="*", label=f"End ({outcome})")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Full pipeline-tracking evaluation trajectory -- outcome: {outcome}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    traj_plot_path = os.path.join(EVAL_DIR, "eval_trajectory_3d.png")
    fig.savefig(traj_plot_path, dpi=140)
    plt.close(fig)
    print(f"Saved 3D trajectory plot to {traj_plot_path}")

    # along-track / standoff-gap / attitude vs time, so the whole run
    # (including the two explicit safety checks) is inspectable at a glance.
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(traj["t"], traj["along_track"], label="along-track distance")
    axes[0].axhline(env.total_len, color="k", linestyle="--", label="pipeline length")
    axes[0].set_ylabel("along-track (m)")
    axes[0].legend(fontsize=8)

    axes[1].plot(traj["t"], traj["dist_to_pipe"], color="tab:red", label="distance to pipe")
    axes[1].axhline(env.standoff_target, color="k", linestyle="--", label=f"required min gap ({env.standoff_target:.1f} m)")
    axes[1].set_ylabel("standoff distance (m)")
    axes[1].legend(fontsize=8)

    flip_threshold_deg = float(np.rad2deg(env.flip_threshold))
    axes[2].plot(traj["t"], traj["roll_deg"], label="roll (deg)")
    axes[2].plot(traj["t"], traj["pitch_deg"], label="pitch (deg)")
    axes[2].axhline(flip_threshold_deg, color="k", linestyle="--", linewidth=1)
    axes[2].axhline(-flip_threshold_deg, color="k", linestyle="--", linewidth=1, label=f"flip threshold (+-{flip_threshold_deg:.0f} deg)")
    axes[2].set_ylabel("attitude (deg)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Pipeline tracking over full episode -- outcome: {outcome}, "
                 f"total_reward={total_reward:.1f}")
    fig.tight_layout()
    track_plot_path = os.path.join(EVAL_DIR, "eval_tracking_error.png")
    fig.savefig(track_plot_path, dpi=140)
    plt.close(fig)
    print(f"Saved tracking-error plot to {track_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=6_000_000)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=20,
                         help="number of domain-randomized episodes for the multi-episode success/flip/gap stats")
    args = parser.parse_args()

    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True

    if args.train:
        train(total_timesteps=args.total_timesteps, n_envs=args.n_envs)
    if args.evaluate:
        evaluate(n_eval_episodes=args.eval_episodes)
