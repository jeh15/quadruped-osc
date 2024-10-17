from typing import Optional, Tuple
from absl import app
import os

import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx

# MJX Utilities:
from mujoco.mjx._src import scan
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import Model


def mj_jacobian(
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


def mj_integrate_position(
    model: Model, qpos: jax.Array, qvel: jax.Array, dt: jax.Array,
) -> jax.Array:
    def mjJNT_FREE(
        qpos: jax.Array, qvel: jax.Array, padr: int, vadr: int, dt: jax.Array,
    ) -> jax.Array:
        # Get Slices:
        qpos_t = jax.lax.dynamic_slice(qpos, (padr,), (3,))
        qvel_t = jax.lax.dynamic_slice(qvel, (vadr,), (3,))
        qpos_r = jax.lax.dynamic_slice(qpos, (padr + 3,), (4,))
        qvel_r = jax.lax.dynamic_slice(qvel, (vadr + 3,), (3,))

        # Position Update:
        q = qpos_t + dt * qvel_t

        # Rotation Update:
        w = mjx_math.quat_integrate(
            qpos_r, qvel_r, dt,
        )

        # Return updated qpos:
        return jax.lax.dynamic_update_slice(
            qpos, jnp.concatenate([q, w]), (padr,),
        )

    def mjJNT_BALL(
        qpos: jax.Array, qvel: jax.Array, padr: int, vadr: int, dt: jax.Array,
    ) -> jax.Array:
        # Get slices:
        qpos_r = jax.lax.dynamic_slice(qpos, (padr,), (4,))
        qvel_r = jax.lax.dynamic_slice(qvel, (vadr,), (3,))

        # Update:
        w = mjx_math.quat_integrate(qpos_r, qvel_r, dt)

        # Return updated qpos:
        return jax.lax.dynamic_update_slice(qpos, w, (padr,))

    def mjJNT_SLIDE(
        qpos: jax.Array, qvel: jax.Array, padr: int, vadr: int, dt: jax.Array,
    ) -> jax.Array:
        # Get slices:
        qpos_slice = jax.lax.dynamic_slice(qpos, (padr,), (1,))
        qvel_slice = jax.lax.dynamic_slice(qvel, (vadr,), (1,))

        # Update:
        q = qpos_slice + dt * qvel_slice
        return jax.lax.dynamic_update_slice(qpos, q, (padr,))

    def mjJNT_HINGE(
        qpos: jax.Array, qvel: jax.Array, padr: int, vadr: int, dt: jax.Array,
    ) -> jax.Array:
        # Get slices:
        qpos_slice = jax.lax.dynamic_slice(qpos, (padr,), (1,))
        qvel_slice = jax.lax.dynamic_slice(qvel, (vadr,), (1,))

        # Update:
        q = qpos_slice + dt * qvel_slice
        return jax.lax.dynamic_update_slice(qpos, q, (padr,))

    # Loop over joints:
    def loop(
        carry: jax.Array, xs: Tuple[int, int, int],
    ) -> Tuple[jax.Array, None]:
        qpos = carry
        padr, vadr, jtype = xs
        args = [qpos, qvel, padr, vadr, dt]
        next_qpos = jax.lax.switch(
            jtype,
            [mjJNT_FREE, mjJNT_BALL, mjJNT_SLIDE, mjJNT_HINGE],
            *args,
        )
        return next_qpos, None

    qpos, _ = jax.lax.scan(
        f=loop,
        init=qpos,
        xs=(model.jnt_qposadr, model.jnt_dofadr, model.jnt_type),
    )

    return qpos


def mj_finite_difference_jacobian(
    model: Model,
    d: Data,
    body_ids: jax.Array,
    site_ids: Optional[jax.Array] = None,
    eps: jax.typing.ArrayLike = 1e-6,
) -> jax.Array:
    # Calculate offset:
    if site_ids is not None:
        point = d.site_xpos[site_ids] - d.xpos[body_ids]
    else:
        point = jnp.zeros_like(d.xpos[body_ids])

    # Calculate initial Jacobian:
    jacp, jacr = jax.vmap(mj_jacobian, in_axes=(None, None, 0, 0))(
        model, d, point, body_ids,
    )
    jac = jnp.concatenate([jacp, jacr], axis=-1)

    # Integrate qpos and calculate forward with integrated qpos:
    qpos = mj_integrate_position(model, d.qpos, d.qvel, eps)  # type: ignore
    d = d.replace(qpos=qpos)
    d = mjx.fwd_position(model, d)

    # Calculate new offset:
    if site_ids is not None:
        point = d.site_xpos[site_ids] - d.xpos[body_ids]
    else:
        point = jnp.zeros_like(d.xpos[body_ids])

    # Calculate final Jacobian:
    _jacp, _jacr = jax.vmap(mj_jacobian, in_axes=(None, None, 0, 0))(
        model, d, point, body_ids,
    )
    jac_eps = jnp.concatenate([_jacp, _jacr], axis=-1)

    # Finite Difference Jacobian:
    return jnp.reshape((jac_eps - jac) / eps, shape=(body_ids.shape[0], 6, -1))


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

        # Task Space Jacobians:
        point = data.site_xpos[feet_ids] - data.xpos[calf_ids]
        jacp, jacr = jax.vmap(mj_jacobian, in_axes=(None, None, 0, 0))(
            model, data, point, calf_ids,
        )
        Jv = jnp.reshape(
            jnp.concatenate([jacp, jacr], axis=-1),
            shape=(calf_ids.shape[0], 6, -1),
        )
        dJv = mj_finite_difference_jacobian(model, data, calf_ids, feet_ids)
        Jv_bias = dJv @ data.qvel


if __name__ == '__main__':
    app.run(main)
