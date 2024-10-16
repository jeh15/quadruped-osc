from typing import Tuple
from absl import app
import os

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx

# MJX Utilities:
from mujoco.mjx._src import scan
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import Model


def jac(
    m: Model, d: Data, point: jax.Array, body_id: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Compute pair of (NV, 3) Jacobians of global point attached to body."""
    fn = lambda carry, b: b if carry is None else b + carry
    mask = (jnp.arange(m.nbody) == body_id) * 1
    mask = scan.body_tree(m, fn, 'b', 'b', mask, reverse=True)
    mask = mask[jnp.array(m.dof_bodyid)] > 0

    offset = point - d.subtree_com[jnp.array(m.body_rootid)[body_id]]
    jacp = jax.vmap(lambda a, b=offset: a[3:] + jnp.cross(a[:3], b))(d.cdof)
    jacp = jax.vmap(jnp.multiply)(jacp, mask)
    jacr = jax.vmap(jnp.multiply)(d.cdof[:, :3], mask)

    return jacp, jacr


def main(argv):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx.xml',
    )
    # Mujoco model:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    q_init = jnp.asarray(mj_model.keyframe('home').qpos)
    qd_init = jnp.asarray(mj_model.keyframe('home').qvel)
    default_ctrl = jnp.asarray(mj_model.keyframe('home').ctrl)

    # MJX Model:
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)

    # Initialize State:
    data = data.replace(qpos=q_init, qvel=qd_init, ctrl=default_ctrl)
    data = mjx.forward(model, data)

    # JIT Functions:
    step_fn = jax.jit(mjx.step)

    num_steps = 500
    for i in range(num_steps):
        data = data.replace(ctrl=default_ctrl)
        data = step_fn(model, data)

        # Dynamics Equation: M @ dv + C = B @ u + J.T @ f
        # Task Space  Objective: ddx_task = vJ @ dv + vJ_bias

        # Mass Matrix:
        M = data.qM

        # Coriolis and Gravity:
        C = data.qfrc_bias

        # Contact Jacobian:
        J = data.efc_J[data.contact.efc_address]

        # Jacobian:
        vJ = jac(model, data, jnp.zeros(3), 1)


if __name__ == '__main__':
    app.run(main)
