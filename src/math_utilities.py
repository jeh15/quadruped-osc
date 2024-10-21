from typing import Tuple
import functools

import jax
import jax.numpy as jnp


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

    return jacp.T, jacr.T


def mj_jacobian_dot(
    m: Model,
    d: Data,
    point: jax.Array,
    body_id: jax.Array,
) -> jax.Array:
    """Computes derivative of a pair of (NV, 3) Jacobians of global point attached to body."""
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

    return jacp.T, jacr.T
