from typing import Tuple
from absl import app
import os

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx
from brax.mjx import pipeline

from brax.io import mjcf, html

from src.controllers.osc import utilities as osc_utils
from src.controllers.osc import controller
from src.controllers.osc.controller import OSQPConfig

# Types:
from jaxopt.base import KKTSolution
from brax.mjx.base import State
import brax

import pdb


jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_disable_jit', True)


def main(argv):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx_torque.xml',
    )
    # Mujoco model:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    q_init = jnp.asarray(mj_model.keyframe('home').qpos)
    qd_init = jnp.asarray(mj_model.keyframe('home').qvel)
    default_ctrl = jnp.asarray(mj_model.keyframe('home').ctrl)

    feet_site = [
        'front_left_foot',
        'front_right_foot',
        'hind_left_foot',
        'hind_right_foot',
    ]
    feet_site_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        for f in feet_site
    ]
    feet_ids = jnp.asarray(feet_site_ids)

    imu_id = jnp.asarray(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'imu'),
    )

    base_body = [
        'base_link',
    ]
    base_body_id = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, b)
        for b in base_body
    ]
    base_id = jnp.asarray(base_body_id)

    calf_body = [
            'front_left_calf',
            'front_right_calf',
            'hind_left_calf',
            'hind_right_calf',
        ]
    calf_body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, c)
        for c in calf_body
    ]
    calf_ids = jnp.asarray(calf_body_ids)

    # MJX Model:
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)

    # Initialize MJX Data:
    data = data.replace(
        qpos=q_init, qvel=qd_init, ctrl=jnp.zeros_like(default_ctrl),
    )
    data = mjx.forward(model, data)


    # Initialize OSC Controller:
    taskspace_targets = jnp.zeros((5, 6))
    osc_controller = controller.OSCController(
        model=mj_model,
        num_contacts=4,
        num_taskspace_targets=5,
        osqp_config=OSQPConfig(
            tol=1e-3,
            maxiter=4000,
            verbose=False,
            jit=True,
        ),
    )

    # JIT Controller Functions:
    get_data_fn = jax.jit(osc_utils.get_data)
    update_fn = jax.jit(osc_controller.update)
    solve_fn = jax.jit(osc_controller.solve)

    # Initialize Values and Warmstart:
    body_points = jnp.expand_dims(data.site_xpos[imu_id], axis=0)
    feet_points = data.site_xpos[feet_ids]
    points = jnp.concatenate([body_points, feet_points])
    body_ids = jnp.concatenate([base_id, calf_ids])

    osc_data = get_data_fn(model, data, points, body_ids)
    prog_data = update_fn(taskspace_targets, osc_data)

    ddx_desired_jax = jnp.zeros((5, 6))

    Mjax = osc_data.mass_matrix
    Cjax = osc_data.coriolis_matrix
    Jcjax = osc_data.contact_jacobian
    Jjax = osc_data.taskspace_jacobian
    bjax = osc_data.taskspace_bias

    Jjax_p, Jjax_r = jnp.split(Jjax, 2, axis=1)
    Jjax_p = jnp.concatenate(Jjax_p, axis=0)
    Jjax_r = jnp.concatenate(Jjax_r, axis=0)
    Jjax = jnp.concatenate([Jjax_p, Jjax_r], axis=0)

    bjax = jnp.concatenate(bjax, axis=0)

    M = np.loadtxt("debug/M.csv", delimiter=",")
    C = np.loadtxt("debug/C.csv", delimiter=",")
    Jc = np.loadtxt("debug/Jc.csv", delimiter=",")
    J = np.loadtxt("debug/J.csv", delimiter=",")
    b = np.loadtxt("debug/b.csv", delimiter=",")
    ddx_desired = np.loadtxt("debug/ddx_desired.csv", delimiter=",")

    # Simulation comparison:
    print(f"M are equal: {np.allclose(M, Mjax, atol=1e-3)}")
    print(f"C are equal: {np.allclose(C, Cjax, atol=1e-3)}")
    print(f"Jc are equal: {np.allclose(Jc, Jcjax, atol=1e-3)}")
    print(f"J are equal: {np.allclose(J, Jjax, atol=1e-3)}")
    print(f"b are equal: {np.allclose(b, bjax, atol=1e-3)}")

    # Optimization comparison:
    Aeqjax = np.asarray(osc_controller.Aeq_fn(osc_controller.q, osc_data))
    beqjax = np.asarray(-osc_controller.equality_constraints(osc_controller.q, osc_data))
    Aineqjax = np.asarray(osc_controller.Aineq_fn(osc_controller.q))
    bineq = np.asarray(osc_controller.inequality_constraints(osc_controller.q))
    Hjax = np.asarray(osc_controller.H_fn(osc_controller.q, ddx_desired_jax, osc_data))
    fjax = np.asarray(osc_controller.f_fn(osc_controller.q, ddx_desired_jax, osc_data))

    Aeq = np.loadtxt("debug/Aeq.csv", delimiter=",")
    beq = np.loadtxt("debug/beq.csv", delimiter=",")
    Aineq = np.loadtxt("debug/Aineq.csv", delimiter=",")
    bineq = np.loadtxt("debug/bineq.csv", delimiter=",")
    H = np.loadtxt("debug/H.csv", delimiter=",")
    f = np.loadtxt("debug/f.csv", delimiter=",")

    print(f"Aeq are equal: {np.allclose(Aeq, Aeqjax, atol=1e-3)}")
    print(f"beq are equal: {np.allclose(beq, beqjax, atol=1e-3)}")
    print(f"Aineq are equal: {np.allclose(Aineq, Aineqjax, atol=1e-3)}")
    print(f"bineq are equal: {np.allclose(bineq, bineq, atol=1e-3)}")
    print(f"H are equal: {np.allclose(H, Hjax, atol=1e-3)}")
    print(f"f are equal: {np.allclose(f, fjax, atol=1e-3)}")

    pass


if __name__ == '__main__':
    app.run(main)
