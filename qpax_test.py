from typing import Tuple
from absl import app
import os
import functools

import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx
from brax.mjx import pipeline

from src.controllers.osc import utilities as osc_utils
from src.controllers.osc import controller_qpax as controller

# New JAX QP Solver:
import qpax

# Types:
from jaxopt.base import KKTSolution
from brax.mjx.base import State

import time

jax.config.update('jax_enable_x64', True)


def main(argv):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx_torque.xml',
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

    imu_id = jnp.asarray(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'imu'),
    )

    base_body = [
        'base_link',
    ]
    base_body_id = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, b)
        for b in base_body
    ]
    base_id = jnp.asarray(base_body_id)

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

    # Initialize MJX Data:
    data = data.replace(
        qpos=q_init, qvel=qd_init, ctrl=jnp.zeros_like(default_ctrl),
    )
    data = mjx.forward(model, data)

    # JIT Functions:
    env_init = functools.partial(
        pipeline.init,
        act=None,
        ctrl=None,
        unused_debug=False,
    )
    env_step = functools.partial(
        pipeline.step,
        unused_debug=False,
    )
    init_fn = jax.jit(jax.vmap(env_init, in_axes=(None, 0, 0)))
    step_fn = jax.jit(jax.vmap(env_step, in_axes=(None, 0, 0)))

    # Number of Parallel Envs:
    batch_size = 64

    # Initialize OSC Controller:
    taskspace_targets = jnp.zeros((batch_size, 5, 6))
    osc_controller = controller.OSCController(
        model=mj_model,
        num_contacts=4,
        num_taskspace_targets=5,
    )

    # JIT Controller Functions:
    get_data_fn = jax.jit(jax.vmap(osc_utils.get_data, in_axes=(None, 0, 0, None)))
    update_fn = jax.jit(jax.vmap(osc_controller.update, in_axes=(0, 0)))

    q_init = jnp.tile(q_init, (batch_size, 1))
    qd_init = jnp.tile(qd_init, (batch_size, 1))
    state = init_fn(model, q_init, qd_init)

    # Initialize Values and Warmstart:
    num_steps = 10000
    body_ids = jnp.concatenate([base_id, calf_ids])

    # QPAX Functions:
    batch_solve = jax.vmap(qpax.solve_qp_primal, in_axes=(0, 0, 0, 0, 0, 0))

    def loop(
        carry: Tuple[State, KKTSolution, jax.Array], xs: jax.Array,
    ) -> Tuple[Tuple[State, KKTSolution, jax.Array], State]:
        state, key = carry
        key, subkey = jax.random.split(key)

        # Get Body and Feet Points:
        body_points = jnp.expand_dims(state.site_xpos[:, imu_id], axis=1)
        feet_points = state.site_xpos[:, feet_ids]
        points = jnp.concatenate([body_points, feet_points], axis=1)

        # Get OSC Data
        osc_data = get_data_fn(model, state, points, body_ids)

        # Update QP:
        prog_data = update_fn(taskspace_targets, osc_data)

        # Solve QP:
        xs = batch_solve(prog_data.H, prog_data.f, prog_data.A, prog_data.b, prog_data.G, prog_data.h)
        dv, u, z = jnp.split(xs, [osc_controller.dv_idx, osc_controller.u_idx], axis=-1)

        # Step Simulation:
        state = step_fn(model, state, u)

        return (state, subkey), state

    # Run Loop:
    initial_state = init_fn(model, q_init, qd_init)
    key = jax.random.key(0)
    start_time = time.time()
    (final_state, _), states = jax.lax.scan(
        f=loop,
        init=(initial_state, key),
        xs=jnp.arange(num_steps),
    )
    final_state.q.block_until_ready()
    print(f"Time Elapsed: {time.time() - start_time}")


if __name__ == '__main__':
    app.run(main)
