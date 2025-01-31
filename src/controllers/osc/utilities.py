import numpy as np
import numpy.typing as npt

from flax import struct

import mujoco

from brax.base import System
from brax.mjx.base import State
from mujoco.mjx._src.types import Model, Data


@struct.dataclass
class OSCData:
    mass_matrix: npt.ArrayLike
    coriolis_matrix: npt.ArrayLike
    contact_jacobian: npt.ArrayLike
    taskspace_jacobian: npt.ArrayLike
    taskspace_bias: npt.ArrayLike
    contact_mask: npt.ArrayLike
    previous_q: npt.ArrayLike
    previous_qd: npt.ArrayLike


def get_data(
    model: Model | System,
    data: Data | State,
    points: npt.ArrayLike,
    body_ids: npt.ArrayLike,
) -> OSCData:
    """Get OSC data from Mujoco Model and Data."""
    # Mass Matrix:
    mass_matrix = data.qM

    # Coriolis Matrix:
    coriolis_matrix = data.qfrc_bias

    # Taskspace Jacobians:

    mujoco.mj_jac(
        model,
        data,
        jacp,
        jacr,
        points,
        body_ids,
    )

    mujoco.mj_jacDot(
        model,
        data,
        jacp_dot,
        jacr_dot,
        points,
        body_ids,
    )

    # Taskspace Jacobian -> Shape: (num_body_ids, 6, NV)
    taskspace_jacobian = np.concatenate([jacp, jacr], axis=-2)

    # Jacobian Dot -> Shape: (num_body_ids, 6, NV)
    jacobian_dot = np.concatenate([jacp_dot, jacr_dot], axis=-2)

    # Taskspace Bias Acceleration -> Shape: (num_body_ids, 6)
    taskspace_bias = jacobian_dot @ data.qvel

    # Contact Jacobian -> Shape: (num_contacts, NV, 3) -> (NV, 3 * num_contacts)
    # TODO(jeh15) Translational only:
    contact_jacobian = np.concatenate(
        jax.vmap(jnp.transpose)(taskspace_jacobian[1:])[:, :, :3],
        axis=-1,
    )

    # Contact Mask: (Less than milimeter)
    contact_mask = data.contact.dist <= 1e-3

    # Pack Struct:
    return OSCData(
        mass_matrix=mass_matrix,
        coriolis_matrix=coriolis_matrix,
        contact_jacobian=contact_jacobian,
        taskspace_jacobian=taskspace_jacobian,
        taskspace_bias=taskspace_bias,
        contact_mask=contact_mask,
        previous_q=data.qpos,
        previous_qd=data.qvel,
    )
