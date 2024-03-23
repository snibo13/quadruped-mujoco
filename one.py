# one.py

import mujoco
import os
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


m = mujoco.MjModel.from_xml_path(
    os.path.join("/home/sidney/mujoco-3.1.3/model/", "humanoid/humanoid.xml")
)
d = mujoco.MjData(m)
model = m
data = d

best_offset = -0.0005070000000000001

# Finding the equilibrium forces.
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
data.qacc = 0
data.qpos[2] += best_offset
qpos0 = data.qpos.copy()  # Save the position setpoint.
mujoco.mj_inverse(model, data)
qfrc0 = data.qfrc_inverse.copy()

# Find the control signal that generates the equilibrium forces.
ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.

# LQR
nu = model.nu
R = np.eye(nu)

nv = model.nv  # DoFs

# Balance cost

# Compute the Jacobian of the dynamics.
mujoco.mj_resetData(model, data)
data.qpos = qpos0
mujoco.mj_forward(model, data)
jac_com = np.zeros((3, nv))
mujoco.mj_jacSubtreeCom(model, data, jac_com, m.body("torso").id)

jac_foot = np.zeros((3, nv))
mujoco.mj_jacSubtreeCom(model, data, jac_foot, m.body("foot_left").id)

jac_diff = jac_com - jac_foot
Qbalance = jac_diff.T @ jac_diff


# Displacement penalty
# Get all joint names.
joint_names = [model.joint(i).name for i in range(model.njnt)]

# Get indices into relevant sets of joints.
root_dofs = range(6)
body_dofs = range(6, nv)
abdomen_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if "abdomen" in name and not "z" in name
]
left_leg_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if "left" in name
    and ("hip" in name or "knee" in name or "ankle" in name)
    and not "z" in name
]
balance_dofs = abdomen_dofs + left_leg_dofs
other_dofs = np.setdiff1d(body_dofs, balance_dofs)

# Cost coefficients.
BALANCE_COST = 1000  # Balancing.
BALANCE_JOINT_COST = 3  # Joints required for balancing.
OTHER_JOINT_COST = 0.3  # Other joints.

# Construct the Qjoint matrix.
Qjoint = np.eye(nv)
Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

# Construct the Q matrix for position DoFs.
Qpos = BALANCE_COST * Qbalance + Qjoint

# No explicit penalty for velocities.
Q = np.block([[Qpos, np.zeros((nv, nv))], [np.zeros((nv, 2 * nv))]])

# Set the initial state and control.
mujoco.mj_resetData(model, data)
data.ctrl = ctrl0
data.qpos = qpos0

# Allocate the A and B matrices, compute them.
A = np.zeros((2 * nv, 2 * nv))
B = np.zeros((2 * nv, nu))
epsilon = 1e-6
flg_centered = True
mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

# Reset before sim
mujoco.mj_resetData(model, data)
data.qpos = qpos0
data.ctrl = ctrl0

dq = np.zeros(nv)
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.lookat = data.body("torso").subtree_com
    viewer.cam.distance = 3

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # Compute the control signal.
        mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T

        d.ctrl = ctrl0 - K @ dx

        # Step the simulation.
        mujoco.mj_step(m, d)

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
