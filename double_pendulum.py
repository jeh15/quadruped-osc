from typing import Optional, Tuple
from absl import app
import os
import functools

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx

from src.math_utilities import mj_jacobian, mj_jacobian_dot


jax.config.update('jax_enable_x64', True)
jax.config.update('jax_disable_jit', False)


def main(argv=None):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/double_pendulum/double_pendulum.xml',
    )
    # Mujoco model:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    q_init = jnp.array([0.0, 0.0])
    qd_init = jnp.array([0.0, 0.0])

    end_effector_id = np.array([
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'end_effector'),
    ])
    link_2_id = np.array([
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'link_2'),
    ])

    # MJX Model:
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)

    # Initialize MJX Data:
    data = data.replace(qpos=q_init, qvel=qd_init, ctrl=np.zeros(model.nu))

    # Initialize Mujoco Data:
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos = q_init
    mj_data.qvel = qd_init
    mj_data.ctrl = np.zeros(model.nu)

    # Minimal Pipeline Steps:
    data = mjx.kinematics(model, data)
    data = mjx.com_pos(model, data)

    # Calculate Jacobian:
    point = data.site_xpos[end_effector_id][0]
    body_id = link_2_id[0]
    jacp, jacr = mj_jacobian(model, data, jnp.squeeze(point), body_id)
    J_mjx = np.concatenate([jacp, jacr])

    # Minimal Mujoco Pipeline:
    mujoco.mj_kinematics(mj_model, mj_data)
    mujoco.mj_comPos(mj_model, mj_data)

    # Mujoco:
    jp = np.zeros((3, mj_model.nv))
    jr = np.zeros((3, mj_model.nv))
    mujoco.mj_jac(mj_model, mj_data, jp, jr, np.squeeze(point), body_id)
    J_mujoco = np.concatenate([jp, jr])

    # Hand Rolled Jacobian:
    q = data.qpos
    L1 = 0.3
    L2 = 0.3

    J_hand = np.array([
        [L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]), L2 * np.cos(q[0] + q[1])],
        [-L1 * np.sin(q[0]) - L2 * np.sin(q[0] + q[1]), -L2 * np.sin(q[0] + q[1])],
    ])

    J_hand_ = np.array([
        [-L1 * np.sin(q[0]) - L2 * np.sin(q[0] + q[1]), -L2 * np.sin(q[0] + q[1])],
        [L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]), L2 * np.cos(q[0] + q[1])],
    ])

    pass


if __name__ == '__main__':
    app.run(main)
