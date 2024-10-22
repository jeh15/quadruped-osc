from typing import Tuple
from absl import app
import os

import time

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx
from brax.mjx import pipeline

from src.controllers.osc import utilities as osc_utils
from src.controllers.osc import controller

jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_disable_jit', True)


def main(argv):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx.xml',
    )
    # Mujoco model:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    q_init = jnp.asarray(mj_model.keyframe('home').qpos)
    qd_init = np.asarray(mj_model.keyframe('home').qvel)
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
    data = data.replace(qpos=q_init, qvel=qd_init, ctrl=default_ctrl)
    data = mjx.forward(model, data)

    # JIT Functions:
    init_fn = jax.jit(pipeline.init)
    step_fn = jax.jit(pipeline.step)

    # Initialize OSC Controller:
    taskspace_targets = jnp.zeros((5, 6))
    osc_controller = controller.OSCController(
        model=model,
        num_contacts=4,
        num_taskspace_targets=5,
        use_motor_model=True,
    )

    # JIT Controller Functions:
    get_data_fn = osc_utils.get_data
    update_fn = osc_controller.update
    solve_fn = osc_controller.solve

    state = init_fn(model, q=q_init, qd=qd_init, ctrl=default_ctrl)
    for _ in range(1000):
        body_points = jnp.zeros((1, 3))
        feet_points = (data.site_xpos[feet_ids] - data.xpos[calf_ids])
        points = jnp.concatenate([body_points, feet_points])
        body_ids = jnp.concatenate([base_id, calf_ids])

        osc_data = get_data_fn(model, state, points, body_ids)

        prog_data = update_fn(taskspace_targets, osc_data)

        solution = solve_fn(prog_data)

        state = step_fn(model, state, default_ctrl)

    # def loop(carry: jax.Array, xs: None) -> Tuple[jax.Array, None]:
    #     state = carry

    #     body_points = jnp.zeros((1, 3))
    #     feet_points = (data.site_xpos[feet_ids] - data.xpos[calf_ids])
    #     points = jnp.concatenate([body_points, feet_points])
    #     body_ids = jnp.concatenate([base_id, calf_ids])

    #     osc_data = get_data_fn(model, state, points, body_ids)

    #     prog_data = update_fn(taskspace_targets, osc_data)

    #     solution = solve_fn(prog_data)

    #     state = step_fn(model, state, default_ctrl)

    #     return (state), None

    # # Run Loop:
    # initial_state = init_fn(model, q=q_init, qd=qd_init, ctrl=default_ctrl)
    # start_time = time.time()
    # final_state, osc_data = jax.lax.scan(
    #     f=loop,
    #     init=initial_state,
    #     xs=None,
    #     length=1000,
    # )

    print(f"Time: {time.time() - start_time}")


if __name__ == '__main__':
    app.run(main)
