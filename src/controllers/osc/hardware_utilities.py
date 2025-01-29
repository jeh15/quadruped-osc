from unitree_bindings import unitree_api

import jax
import jax.numpy as jnp
from flax import struct

import numpy as np

import mujoco.mjx as mjx

from brax.base import System
from brax.mjx.base import State
from mujoco.mjx._src.types import Model, Data

import src.math_utilities as math_utils
from src.controllers.osc.utilities import OSCData


# Utility to convert from robot -> mujoco model and back:
convert_motor_idx = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
convert_foot_idx = np.array([1, 0, 3, 2])

@struct.dataclass
class RobotState:
    motor_position: jax.Array
    motor_velocity: jax.Array
    motor_acceleration: jax.Array
    torque_estimate: jax.Array
    body_rotation: jax.Array
    body_velocity: jax.Array
    body_acceleration: jax.Array
    foot_contacts: jax.Array


def get_robot_state(
    unitree: unitree_api.MotorController,
) -> RobotState:
    # Get Motor State:
    motor_state = unitree.get_motor_state()

    # Get IMU Data:
    imu_state = unitree.get_imu_state()

    # Get Low State:
    low_state = unitree.get_low_state()

    # Reorder states to match mj model:
    """
        mj_model : [FL FR HL HR]
        robot : [FR FL HR HL]
    """
    motor_position = np.asarray(motor_state.q)[convert_motor_idx]
    motor_velocity = np.asarray(motor_state.qd)[convert_motor_idx]
    motor_acceleration = np.asarray(motor_state.qdd)[convert_motor_idx]
    torque_estimate = np.asarray(motor_state.torque_estimate)[convert_motor_idx]
    body_rotation = np.asarray(imu_state.quaternion)
    body_velocity = np.asarray(imu_state.gyroscope)
    body_acceleration = np.asarray(imu_state.accelerometer)
    foot_contacts = np.asarray(low_state.foot_force)[convert_foot_idx]

    return RobotState(
        motor_position=motor_position,
        motor_velocity=motor_velocity,
        motor_acceleration=motor_acceleration,
        torque_estimate=torque_estimate,
        body_rotation=body_rotation,
        body_velocity=body_velocity,
        body_acceleration=body_acceleration,
        foot_contacts=foot_contacts,
    )

def get_data(
    model: Model | System,
    data: Data | State,
    robot_state: RobotState,
    points: jax.Array,
    body_ids: jax.Array,
    assume_contact: bool = False,
) -> OSCData:
    """Update MJX Model and Data or Brax System and State (MJX Pipeline)."""
    # Update Data:
    q = jnp.concatenate(
        [jnp.array([0.0, 0.0, 0.0]), robot_state.body_rotation, robot_state.motor_position]
    )
    qd = jnp.concatenate(
        [jnp.array([0.0, 0.0, 0.0]), robot_state.body_velocity, robot_state.motor_velocity]
    )
    data = data.replace(
        qpos=q, qvel=qd, ctrl=jnp.zeros(shape=(12,)),
    )

    # Forward Dynamics:
    data = mjx.forward(model, data)

    """Get OSC data from MJX Model and Data or Brax System and State (MJX Pipeline)."""
    # Mass Matrix:
    mass_matrix = data.qM

    # Coriolis Matrix:
    coriolis_matrix = data.qfrc_bias

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

    # Contact Jacobian -> Shape: (num_contacts, NV, 3) -> (NV, 3 * num_contacts)
    # TODO(jeh15) Translational only:
    contact_jacobian = jnp.concatenate(
        jax.vmap(jnp.transpose)(taskspace_jacobian[1:])[:, :, :3],
        axis=-1,
    )

    # Contact Mask: (Force threshold) -- Default no load value ~ 17.0 - 18.0
    contact_mask = robot_state.foot_contacts <= 24.0
    contact_mask = jax.lax.select(assume_contact, jnp.ones_like(contact_mask), contact_mask)

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


def update_mj_data(
    model: Model | System,
    data: Data | State,
    robot_state: RobotState,
) -> Data:
    """Update MJX Model and Data or Brax System and State (MJX Pipeline)."""
    # Update Data:
    q = jnp.concatenate(
        [jnp.array([0.0, 0.0, 0.0]), robot_state.body_rotation, robot_state.motor_position]
    )
    qd = jnp.concatenate(
        [jnp.array([0.0, 0.0, 0.0]), robot_state.body_velocity, robot_state.motor_velocity]
    )
    data = data.replace(
        qpos=q, qvel=qd, ctrl=jnp.zeros(shape=(12,)),
    )

    # Forward Dynamics:
    data = mjx.forward(model, data)

    return data


# def get_data(
#     model: Model | System,
#     data: Data | State,
#     robot_state: RobotState,
#     points: jax.Array,
#     body_ids: jax.Array,
#     ignore_contacts: bool = False,
# ) -> OSCData:
#     """Get OSC data from MJX Model and Data or Brax System and State (MJX Pipeline)."""
#     # Mass Matrix:
#     mass_matrix = data.qM

#     # Coriolis Matrix:
#     coriolis_matrix = data.qfrc_bias

#     # Taskspace Jacobian:
#     jacp_dot, jacr_dot = jax.vmap(
#         math_utils.mj_jacobian_dot, in_axes=(None, None, 0, 0), out_axes=(0, 0),
#     )(model, data, points, body_ids)

#     # Jacobian Dot -> Shape: (num_body_ids, 6, NV)
#     jacobian_dot = jnp.concatenate([jacp_dot, jacr_dot], axis=-2)

#     # Taskspace Bias Acceleration -> Shape: (num_body_ids, 6)
#     taskspace_bias = jacobian_dot @ data.qvel
#     jacp, jacr = jax.vmap(
#         math_utils.mj_jacobian, in_axes=(None, None, 0, 0), out_axes=(0, 0),
#     )(model, data, points, body_ids)

#     # Taskspace Jacobian -> Shape: (num_body_ids, 6, NV)
#     taskspace_jacobian = jnp.concatenate([jacp, jacr], axis=-2)

#     # Contact Jacobian -> Shape: (num_contacts, NV, 3) -> (NV, 3 * num_contacts)
#     # TODO(jeh15) Translational only:
#     contact_jacobian = jnp.concatenate(
#         jax.vmap(jnp.transpose)(taskspace_jacobian[1:])[:, :, :3],
#         axis=-1,
#     )

#     # Contact Mask: (Force threshold) -- Default no load value ~ 17.0 - 18.0
#     contact_mask = robot_state.foot_contacts <= 24.0
#     contact_mask = jax.lax.select(ignore_contacts, jnp.ones_like(contact_mask), contact_mask)

#     # Pack Struct:
#     return OSCData(
#         mass_matrix=mass_matrix,
#         coriolis_matrix=coriolis_matrix,
#         contact_jacobian=contact_jacobian,
#         taskspace_jacobian=taskspace_jacobian,
#         taskspace_bias=taskspace_bias,
#         contact_mask=contact_mask,
#         previous_q=data.qpos,
#         previous_qd=data.qvel,
#     )
