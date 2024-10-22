import jax
import jax.numpy as jnp
from flax import struct

from brax.base import System
from brax.mjx.base import State
from mujoco.mjx._src.types import Model, Data

import mujoco

import src.math_utilities as math_utils


@struct.dataclass
class OSCData:
    mass_matrix: jax.Array
    coriolis_matrix: jax.Array
    contact_jacobian: jax.Array
    taskspace_jacobian: jax.Array
    taskspace_bias: jax.Array
    previous_q: jax.Array
    previous_qd: jax.Array


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


def get_data(
    model: Model | System,
    data: Data | State,
    points: jax.Array,
    body_ids: jax.Array,
) -> OSCData:
    """Get OSC data from MJX Model and Data or Brax System and State (MJX Pipeline)."""
    # Mass Matrix:
    mass_matrix = data.qM

    # Coriolis Matrix:
    coriolis_matrix = data.qfrc_bias

    # Mujoco Contact Jacobian:
    # if model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
    #     pyramid_jacobians = jnp.array(
    #         jnp.split(data.efc_J, data.contact.efc_address)[1:],
    #     )
    #     basis = jax.vmap(_create_basis)(data.contact.friction)
    #     contact_jacobians = basis @ pyramid_jacobians
    #     contact_jacobian = jnp.concatenate(contact_jacobians)
    # elif model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
    #     contact_jacobians = jnp.array(
    #         jnp.split(data.efc_J, data.contact.efc_address)[1:],
    #     )
    #     contact_jacobian = jnp.concatenate(contact_jacobians)
    # else:
    #     raise ValueError("Invalid Cone Type.")

    # Taskspace Jacobian:
    jacp_dot, jacr_dot = jax.vmap(
        math_utils.mj_jacobian_dot, in_axes=(None, None, 0, 0), out_axes=(0, 0),
    )(model, data, points, body_ids)

    # Jacobian Dot -> Shape: (num_body_ids, 6, NV)
    jacobian_dot = jnp.concatenate([jacp_dot, jacr_dot], axis=-2)

    # Taskspace Bias Acceleration -> Shape: (num_body_ids, 6)
    taskspace_bias = jacobian_dot @ data.qvel
    jacp, jacr = jax.vmap(
        math_utils.mj_jacobian, in_axes=(None, None, 0, 0), out_axes=(0, 0),
    )(model, data, points, body_ids)

    # Taskspace Jacobian -> Shape: (num_body_ids, 6, NV)
    taskspace_jacobian = jnp.concatenate([jacp, jacr], axis=-2)

    # Contact Jacobian -> Shape: (num_contacts, NV, 6) -> (NV, 6 * num_contacts)
    contact_jacobian = jnp.concatenate(
        jax.vmap(jnp.transpose)(taskspace_jacobian[1:]),
        axis=-1,
    )

    # Pack Struct:
    return OSCData(
        mass_matrix=mass_matrix,
        coriolis_matrix=coriolis_matrix,
        contact_jacobian=contact_jacobian,
        taskspace_jacobian=taskspace_jacobian,
        taskspace_bias=taskspace_bias,
        previous_q=data.qpos,
        previous_qd=data.qvel,
    )
