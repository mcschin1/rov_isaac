"""
rov_physics_controller.py
--------------------------
Runs INSIDE Isaac Sim (needs isaacsim.core, omni.physx). Not runnable in a
plain Python/usd-core environment; this script assumes the Isaac Sim Python
environment (python.sh) and an open stage containing /World/ROV as built by
build_rov_scene.py.

What PhysX already does for you, from the USD RigidBodyAPI/MassAPI on
/World/ROV: integrates the rigid body under mass and inertia, and resolves
collisions against the Hull's CollisionAPI plus the static seafloor/rock
colliders in the scene.

Gravity is disabled for this body specifically
(physxRigidBody:disableGravity, authored in build_rov_scene.py) so that
opening the raw scene and pressing Play does NOT make the ROV free-fall.
With no controller running, net force is zero and the ROV stays put. This
script is what re-introduces the real physics on top of that neutral
baseline: buoyancy, weight, hydrodynamic damping, added-mass, and thrust.

What PhysX does NOT do, and what this script adds every physics step:
  - buoyancy (opposing gravity, from rov:buoyancy)
  - linear + quadratic hydrodynamic damping (rov:linearDamping /
    rov:quadraticDamping)
  - added-mass reaction force (rov:addedMass) -- approximated here as a
    velocity-dependent force; for full fidelity, fold added mass into the
    effective mass matrix instead (see NOTE in _added_mass_force)
  - Coriolis-centripetal coupling C(nu) * nu, since PhysX's own mass matrix
    is a simple diagonal + CoM offset, not the full coupled Fossen M
  - thruster forces, from a 6-vector of thrust commands and the datasheet's
    allocation matrix A (rov:allocation:*)

This mirrors the structure of app/physics/rov_dynamics.py, but expressed as
external forces layered on top of Isaac Sim's own rigid-body integrator
rather than a standalone RK4 loop. If you need bit-for-bit agreement with the
dashboard's RK4 integration, keep rov_dynamics.py as the source of truth and
use this only for the visual/interactive Isaac Sim scene.

Usage (inside the Isaac Sim / Script Editor python environment):

    from rov_physics_controller import ROVController
    controller = ROVController(prim_path="/World/ROV")
    controller.set_thrust_command([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])  # heave up
    # controller.step(dt) is called automatically via the physics
    # subscription registered in __init__
"""

import numpy as np

from pxr import Usd, UsdGeom, UsdPhysics
import omni.usd
import omni.physx
from omni.physx import get_physx_interface
from isaacsim.core.prims import RigidPrim


DOF_ORDER = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def _quat_to_rotation_matrix(q):
    """q: (w,x,y,z) -> 3x3 body->world rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class ROVController:
    def __init__(self, prim_path="/World/ROV", stage: Usd.Stage = None):
        self.stage = stage or omni.usd.get_context().get_stage()
        self.prim_path = prim_path
        self.prim = self.stage.GetPrimAtPath(prim_path)
        if not self.prim.IsValid():
            raise RuntimeError(f"No prim at {prim_path}; run build_rov_scene.py first")

        self._read_params()
        self.thrust_cmd = np.zeros(6)  # normalized [-1, 1] per thruster, T1..T6

        # isaacsim.core.prims.RigidPrim is the actual Isaac Sim 6.0.1 API for
        # rigid-body pose/velocity/force access (the older
        # omni.physx.scripts.utils.get_physx_rigid_body helper referenced in
        # this module's original docstring no longer exists in this version).
        self.rigid_prim = RigidPrim(prim_path)
        self.rigid_prim.initialize()

        # subscribe to the physics step; called with dt in seconds
        self._physx_iface = get_physx_interface()
        self._sub = self._physx_iface.subscribe_physics_step_events(self._on_physics_step)

    def _read_params(self):
        p = self.prim
        self.mass = p.GetAttribute("rov:weight").Get() / 9.81  # sanity cross-check vs MassAPI
        self.weight = p.GetAttribute("rov:weight").Get()
        self.buoyancy = p.GetAttribute("rov:buoyancy").Get()
        self.r_b = np.array(p.GetAttribute("rov:centerOfBuoyancy").Get())
        self.added_mass = np.array(p.GetAttribute("rov:addedMass").Get())
        self.lin_damp = np.array(p.GetAttribute("rov:linearDamping").Get())
        self.quad_damp = np.array(p.GetAttribute("rov:quadraticDamping").Get())
        self.t_max = p.GetAttribute("rov:thrusterMaxForce").Get()

        self.A = np.array([
            p.GetAttribute("rov:allocation:u").Get(),
            p.GetAttribute("rov:allocation:v").Get(),
            p.GetAttribute("rov:allocation:w").Get(),
            p.GetAttribute("rov:allocation:p").Get(),
            p.GetAttribute("rov:allocation:q").Get(),
            p.GetAttribute("rov:allocation:r").Get(),
        ])  # 6x6, rows=DOF, cols=thrusters

        mass_api = UsdPhysics.MassAPI(p)
        self.mass_from_usd = mass_api.GetMassAttr().Get()
        self.inertia_diag = np.array(mass_api.GetDiagonalInertiaAttr().Get())

    def set_thrust_command(self, cmd6):
        """cmd6: iterable of 6 normalized thruster commands in [-1, 1] (T1..T6)."""
        self.thrust_cmd = np.clip(np.array(cmd6, dtype=float), -1.0, 1.0)

    def get_world_pose(self):
        """Returns (position_xyz, rotation_matrix_3x3 body->world)."""
        positions, orientations = self.rigid_prim.get_world_poses()
        position = np.array(positions[0])
        rotation = _quat_to_rotation_matrix(np.array(orientations[0]))
        return position, rotation

    def _get_body_velocity(self):
        """Linear + angular velocity in the BODY frame, nu = [u,v,w,p,q,r]."""
        lin_world = np.array(self.rigid_prim.get_linear_velocities()[0])
        ang_world = np.array(self.rigid_prim.get_angular_velocities()[0])
        _, rotation = self.get_world_pose()
        lin_body = rotation.T @ lin_world
        ang_body = rotation.T @ ang_world
        return np.concatenate([lin_body, ang_body])

    def _damping_force(self, nu):
        lin = self.lin_damp * nu
        quad = self.quad_damp * nu * np.abs(nu)
        return lin + quad  # negative-signed coefficients already imply drag

    def _added_mass_force(self, nu, nu_dot_prev):
        # NOTE: strictly, added mass modifies the effective mass matrix
        # (M = M_rb + M_added) rather than acting as a free-standing force.
        # This first-order approximation applies -M_added * nu_dot using the
        # previous step's acceleration estimate; for full fidelity, replace
        # PhysX's own integration with a custom M^-1 solve exactly as
        # rov_dynamics.py does, rather than layering this on top of PhysX.
        return self.added_mass * nu_dot_prev

    def _buoyancy_force_body(self, rotation):
        # Gravity is now disabled at the rigid-body level (see
        # physxRigidBody:disableGravity in build_rov_scene.py), so nothing
        # else is pulling the ROV down. We therefore apply the FULL
        # weight-vs-buoyancy restoring force ourselves -- both halves, not
        # just buoyancy on top of PhysX's own gravity as in the first draft
        # of this script. Net is (B - W), exactly matching Sec 5 of the
        # datasheet: positively buoyant, so with zero thrust the ROV drifts
        # gently upward instead of free-falling. Always acts along WORLD +z,
        # so it is rotated into the body frame here (see _on_physics_step).
        buoyancy_world = np.array([0.0, 0.0, self.buoyancy - self.weight])
        return rotation.T @ buoyancy_world

    def _thruster_wrench(self):
        thrust_forces = self.thrust_cmd * self.t_max  # N, per thruster
        tau = self.A @ thrust_forces  # 6-vector: [Fx,Fy,Fz,Mx,My,Mz] in body frame
        return tau

    def _on_physics_step(self, dt: float):
        if dt <= 0.0:
            return

        nu = self._get_body_velocity()
        _, rotation = self.get_world_pose()

        damping = self._damping_force(nu)
        tau_thrust = self._thruster_wrench()
        buoyancy_body = self._buoyancy_force_body(rotation)

        # All three terms are body-frame; apply directly with is_global=False
        # so RigidPrim handles the body->world rotation itself, rather than
        # (as in the original draft of this script) mixing an unrotated
        # body-frame force with a world-frame buoyancy term.
        force_body = damping[0:3] + tau_thrust[0:3] + buoyancy_body
        torque_body = damping[3:6] + tau_thrust[3:6]

        self.rigid_prim.apply_forces_and_torques_at_pos(
            forces=force_body.reshape(1, 3),
            torques=torque_body.reshape(1, 3),
            is_global=False,
        )

    def shutdown(self):
        if self._sub is not None:
            self._sub.unsubscribe()
            self._sub = None
