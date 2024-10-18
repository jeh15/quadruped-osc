from typing import Optional, Tuple
from absl import app
import os

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


jax.config.update('jax_disable_jit', False)


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

    # Statically known function list:
    if model.njnt >= 7:
        function_list = [mjJNT_FREE, mjJNT_BALL, mjJNT_SLIDE, mjJNT_HINGE]
        jtype_offset = 0
    elif model.njnt >= 4:
        function_list = [mjJNT_BALL, mjJNT_SLIDE, mjJNT_HINGE]
        jtype_offset = 1
    else:
        function_list = [mjJNT_SLIDE, mjJNT_HINGE]
        jtype_offset = 2

    def loop(
        carry: jax.Array, xs: Tuple[int, int, int],
    ) -> Tuple[jax.Array, None]:
        qpos = carry
        padr, vadr, jtype = xs
        args = [qpos, qvel, padr, vadr, dt]
        next_qpos = jax.lax.switch(
            jtype - jtype_offset,
            function_list,
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


def mj_jacobian_dot(
    m: Model,
    d: Data,
    point: jax.Array,
    body_id: jax.Array,
) -> jax.Array:
    """Compute pair of (NV, 3) Jacobians of global point attached to body."""
    fn = lambda carry, b: b if carry is None else b + carry
    mask = (jnp.arange(m.nbody) == body_id) * 1
    mask = scan.body_tree(m, fn, 'b', 'b', mask, reverse=True)
    mask = mask[jnp.array(m.dof_bodyid)] > 0

    offset = point - d.subtree_com[jnp.array(m.body_rootid)[body_id]]

    cdof_dot = d.cdof_dot
    jtype = m.jnt_type
    jnt_fn = lambda carry, jnt: 


    jacp = jax.vmap(lambda a, b=offset: a[3:] + jnp.cross(a[:3], b))(d.cdof)
    jacp = jax.vmap(jnp.multiply)(jacp, mask)
    jacr = jax.vmap(jnp.multiply)(d.cdof[:, :3], mask)

    return jacp, jacr


def main(argv):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/double_pendulum/double_pendulum.xml',
    )
    # Mujoco model:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    q_init = jnp.array([jnp.pi / 2.0])
    qd_init = jnp.array([0.0])
    default_ctrl = jnp.array([1.0])

    mj_data.qpos = q_init
    mj_data.qvel = qd_init
    mj_data.ctrl = default_ctrl

    mujoco.mj_step(mj_model, mj_data)

    jp = np.zeros((3, mj_model.nv))
    jr = np.zeros((3, mj_model.nv))

    ee_site = [
        'end_effector',
    ]
    ee_site_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, e)
        for e in ee_site
    ]
    ee_ids = jnp.asarray(ee_site_ids)

    pendulum_body = [
        'pole',
    ]
    pendulum_body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, p)
        for p in pendulum_body
    ]
    pendulum_ids = jnp.asarray(pendulum_body_ids)

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
        point = data.site_xpos[ee_ids] - data.xpos[pendulum_ids]
        jacp, jacr = jax.vmap(mj_jacobian, in_axes=(None, None, 0, 0))(
            model, data, point, pendulum_ids,
        )
        mj_jacobian_dot(model, data, point, pendulum_ids[0])
        Jv = jnp.reshape(
            jnp.concatenate([jacp, jacr], axis=-1),
            shape=(pendulum_ids.shape[0], 6, -1),
        )
        dJv = mj_finite_difference_jacobian(model, data, pendulum_ids, ee_ids)
        Jv_bias = dJv @ data.qvel

        # Different ways to calculate the Jacobian: (THIS IS THE CORRECT WAY)
        # MJX Jacobian:
        point = data.site_xpos[ee_ids] - data.xpos[pendulum_ids]
        jacp, jacr = jax.vmap(mj_jacobian, in_axes=(None, None, 0, 0))(
            model, data, point, pendulum_ids,
        )
        J_global = jnp.reshape(
            jnp.concatenate([jacp, jacr], axis=-1),
            shape=(pendulum_ids.shape[0], 6, -1),
        )
        # Mujoco Jacobian:
        mujoco.mj_jac(mj_model, mj_data, jp, jr, np.squeeze(point), pendulum_ids[0])
        J_mujoco = np.concatenate([jp, jr])
        # Analytical Jacobian:
        jacp, jacr = jax.vmap(mj_jacobian, in_axes=(None, None, 0, 0))(
            model, data, point, pendulum_ids,
        )
        J_calc = jnp.array(
            [0.6 * jnp.cos(data.qpos[0]), 0.0, -0.6 * jnp.sin(data.qpos[0])]
        )

        # Compare JDot:
        JDot_mjx = mj_finite_difference_jacobian(model, data, pendulum_ids, ee_ids)
        jp = np.zeros((3, mj_model.nv))
        jr = np.zeros((3, mj_model.nv))

        # Pipeline Stages:
        mj_data.ctrl = 10.0
        mujoco.mj_kinematics(mj_model, mj_data)
        mujoco.mj_comPos(mj_model, mj_data)
        mujoco.mj_comVel(mj_model, mj_data)

        point = mj_data.site_xpos[ee_ids] - mj_data.xpos[pendulum_ids]

        mujoco.mj_jacDot(mj_model, mj_data, jp, jr, np.squeeze(point), pendulum_ids[0])
        JDot_mujoco = np.concatenate([jp, jr])

        pass


if __name__ == '__main__':
    app.run(main)
