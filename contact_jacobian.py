from typing import Optional, Tuple
from absl import app
import os
import functools

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx

# MJX Utilities:
from mujoco.mjx._src import scan
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import Model


jax.config.update('jax_enable_x64', True)
jax.config.update('jax_disable_jit', False)


def _create_basis(friction: jax.Array) -> jax.Array:
    basis = jnp.stack([
        jnp.ones((10,)),
        jnp.pad(jnp.array([friction[0], -friction[0]]), (0, 8)),
        jnp.pad(jnp.array([friction[1], -friction[1]]), (2, 6)),
        jnp.pad(jnp.array([friction[2], -friction[2]]), (4, 4)),
        jnp.pad(jnp.array([friction[3], -friction[3]]), (6, 2)),
        jnp.pad(jnp.array([friction[4], -friction[4]]), (8, 0)),
    ])
    return basis


def main(argv):
    pyramid_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx.xml',
    )
    elliptic_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx_elliptic.xml',
    )
    # Mujoco model:
    mj_model_py = mujoco.MjModel.from_xml_path(pyramid_path)
    mj_model_el = mujoco.MjModel.from_xml_path(elliptic_path)
    q_init = jnp.asarray(mj_model_py.keyframe('home').qpos)
    qd_init = np.asarray(mj_model_py.keyframe('home').qvel)
    qd_init[2] = 0.1
    default_ctrl = jnp.asarray(mj_model_py.keyframe('home').ctrl)

    feet_site = [
        'front_left_foot',
        'front_right_foot',
        'hind_left_foot',
        'hind_right_foot',
    ]
    feet_site_ids = [
        mujoco.mj_name2id(mj_model_py, mujoco.mjtObj.mjOBJ_SITE.value, f)
        for f in feet_site
    ]
    feet_ids = jnp.asarray(feet_site_ids)

    calf_body = [
            'front_left_calf',
            'front_right_calf',
            'hind_left_calf',
            'hind_right_calf',
        ]
    calf_body_ids = [
        mujoco.mj_name2id(mj_model_py, mujoco.mjtObj.mjOBJ_BODY.value, c)
        for c in calf_body
    ]
    calf_ids = jnp.asarray(calf_body_ids)

    # MJX Model:
    model = mjx.put_model(mj_model_py)
    data = mjx.make_data(model)

    # Initialize MJX Data:
    data = data.replace(qpos=q_init, qvel=qd_init, ctrl=default_ctrl)

    # Initialize Mujoco Data:
    mj_data = mujoco.MjData(mj_model_el)
    mj_data.qpos = q_init
    mj_data.qvel = qd_init
    mj_data.ctrl = default_ctrl

    """ mj_jacDot Test:"""
    # Minimial MJX Pipeline:
    data = mjx.step(model, data)

    # Minimal Mujoco Pipeline:
    mujoco.mj_step(mj_model_el, mj_data)

    # Compare Jacobian Dot:
    # MJX:
    pyramid_jacobians = jnp.array(
        jnp.split(data.efc_J, data.contact.efc_address)[1:],
    )
    basis = jax.vmap(_create_basis)(data.contact.friction)
    pyramid_jacobians = basis @ pyramid_jacobians
    pyramid_jacobian = jnp.concatenate(pyramid_jacobians)

    # Mujoco:
    elliptic_jacobian = np.reshape(mj_data.efc_J, (-1, mj_model_el.nv))

    point = (data.site_xpos[feet_ids] - data.xpos[calf_ids])[0]
    body_id = calf_ids[0]

    jp = np.zeros((3, mj_model_el.nv))
    jr = np.zeros((3, mj_model_el.nv))
    mujoco.mj_jac(mj_model_el, mj_data, jp, jr, np.squeeze(point), body_id)
    J_mujoco = np.concatenate([jp, jr])

    pass

    # Compare:
    np.isclose(pyramid_jacobian, elliptic_jacobian, atol=1e-6, rtol=1e-6)



if __name__ == '__main__':
    app.run(main)
