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

    # Make sure to transpose these after...
    return jacp, jacr


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

    jtype = m.jnt_type

    def jntid_loop(
        carry: Tuple[jax.Array, int], xs: int,
    ) -> Tuple[Tuple[jax.Array, int], None]:
        def true_fn(jnt_ids, jtype, i):
            jnt_ids = jax.lax.dynamic_update_slice(
                jnt_ids, jnp.repeat(jtype, 2), (i,),
            )
            i += 2
            return jnt_ids, i

        def false_fn(jnt_ids, jtype, i):
            jnt_ids = jax.lax.dynamic_update_slice(
                jnt_ids, jnp.array([jtype]), (i,),
            )
            i += 1
            return jnt_ids, i

        jnt_ids, i = carry
        jtype = xs
        args = [jnt_ids, jtype, i]
        jnt_ids, i = jax.lax.cond(jtype == 0, true_fn, false_fn, *args)

        return (jnt_ids, i), None

    (jntids, _), _ = jax.lax.scan(
        f=jntid_loop,
        init=(jnp.zeros(m.nbody, dtype=jtype.dtype), 0),
        xs=jtype,
    )

    def cvel_loop(
        carry: Tuple[jax.Array, jax.Array, int], xs: int, cvel: jax.Array,
    ) -> Tuple[Tuple[jax.Array, jax.Array, int], None]:
        def true_fn(new_cvel, cdof_mask, i):
            cvel_slice = jax.lax.dynamic_slice(cvel, (i, 0), (3, 6))
            new_cvel = jax.lax.dynamic_update_slice(
                new_cvel, cvel_slice, (i, 6),
            )
            cdof_mask = jax.lax.dynamic_update_slice(
                cdof_mask, jnp.array([1, 1, 1], dtype=jnp.bool), (i,),
            )
            i += 3
            return new_cvel, cdof_mask, i

        def false_fn(new_cvel, cdof_mask, i):
            cvel_slice = jax.lax.dynamic_slice(cvel, (i, 0), (1, 6))
            new_cvel = jax.lax.dynamic_update_slice(
                new_cvel, cvel_slice, (i, 6),
            )
            cdof_mask = jax.lax.dynamic_update_slice(
                cdof_mask, jnp.array([0], dtype=jnp.bool), (i,),
            )
            i += 1
            return new_cvel, cdof_mask, i

        new_cvel, cdof_mask, i = carry
        jtype = xs
        args = [new_cvel, cdof_mask, i]
        new_cvel, cdof_mask, i = jax.lax.cond(
            (jtype == 0) | (jtype == 1), true_fn, false_fn, *args,
        )

        return (new_cvel, cdof_mask, i), None

    (cvel, cdof_mask, _), _ = jax.lax.scan(
        f=functools.partial(cvel_loop, cvel=d.cvel),
        init=(jnp.zeros_like(d.cdof), jnp.zeros(m.nv, dtype=jnp.bool), 0),
        xs=jntids,
    )

    cdof_dot = jax.vmap(lambda x, y: mjx_math.motion_cross(x, y))(cvel, d.cdof)
    cdof_dot = jax.vmap(jnp.multiply)(cdof_dot, cdof_mask)
    _cdof_dot = jax.vmap(jnp.multiply)(d.cdof_dot, (~cdof_mask))
    cdof_dot = cdof_dot + _cdof_dot

    jacp = jax.vmap(lambda a, b=offset: a[3:] + jnp.cross(a[:3], b))(cdof_dot)
    jacp = jax.vmap(jnp.multiply)(jacp, mask)
    jacr = jax.vmap(jnp.multiply)(cdof_dot[:, :3], mask)

    # Make sure to transpose these after...
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

        # Testing this:
        point = data.site_xpos[feet_ids] - data.xpos[calf_ids]
        # jacp, jacr = mj_jacobian_dot(model, data, point[0], calf_ids[0])

        # Mujoco jacDot Test:
        jp = np.zeros((3, mj_model.nv))
        jr = np.zeros((3, mj_model.nv))

        # Make Data:
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos = np.squeeze(data.qpos)
        mj_data.qvel = np.squeeze(data.qvel)
        mj_data.ctrl = np.squeeze(data.ctrl)

        # Step:
        mujoco.mj_step(mj_model, mj_data)
        point = mj_data.site_xpos[feet_site_ids] - mj_data.xpos[calf_body_ids]

        mujoco.mj_jacDot(mj_model, mj_data, jp, jr, np.squeeze(point[0]), calf_body_ids[0])

        new_data = mjx.put_data(mj_model, mj_data)
        jacp, jacr = mj_jacobian_dot(model, new_data, point[0], calf_ids[0])

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

        # Mujoco jacDot Test:
        jp = np.zeros((3, mj_model.nv))
        jr = np.zeros((3, mj_model.nv))

        # Make Data:
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos = np.squeeze(data.qpos)
        mj_data.qvel = np.squeeze(data.qvel)
        mj_data.ctrl = np.squeeze(data.ctrl)

        # Step:
        mujoco.mj_step(mj_model, mj_data)
        point = mj_data.site_xpos[feet_site_ids] - mj_data.xpos[calf_body_ids]

        mujoco.mj_jacDot(mj_model, mj_data, jp, jr, np.squeeze(point[0]), calf_body_ids[0])

        dJv_mujoco = np.concatenate([jp, jr])
        pass


if __name__ == '__main__':
    app.run(main)
