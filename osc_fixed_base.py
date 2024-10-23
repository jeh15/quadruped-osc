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

from brax.io import mjcf, html

from src.controllers.osc import utilities as osc_utils
from src.controllers.osc.tests._fixed_base_controller import OSCController, OSQPConfig

jax.config.update('jax_enable_x64', True)


def main(argv):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/scene_mjx_fixed.xml',
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

    initial_feet_pos = data.site_xpos[feet_ids]

    # JIT Functions:
    init_fn = jax.jit(pipeline.init)
    step_fn = jax.jit(pipeline.step)

    # Initialize OSC Controller:
    taskspace_targets = jnp.zeros((4, 6))
    osc_controller = OSCController(
        model=mj_model,
        num_taskspace_targets=4,
        use_motor_model=False,
        osqp_config=OSQPConfig(
            tol=1e-3,
            maxiter=50000,
            verbose=False,
            jit=True,
        ),
    )

    # JIT Controller Functions:
    get_data_fn = jax.jit(osc_utils.get_data)
    update_fn = jax.jit(osc_controller.update)
    solve_fn = jax.jit(osc_controller.solve)

    state = init_fn(model, q=q_init, qd=qd_init, ctrl=default_ctrl)

    # Initialize Warmstart:
    points = (state.site_xpos[feet_ids] - state.xpos[calf_ids])
    body_ids = calf_ids

    osc_data = get_data_fn(model, state, points, body_ids)

    prog_data = update_fn(taskspace_targets, osc_data)

    init_x = jnp.concatenate([
        jnp.zeros(osc_controller.dv_size),
        jnp.zeros_like(default_ctrl),
    ])

    warmstart = osc_controller.prog.init_params(
        init_x=init_x,
        params_obj=(prog_data.H, prog_data.f),
        params_eq=prog_data.A,
        params_ineq=(prog_data.lb, prog_data.ub),
    )

    state_history = []
    for i in range(1000):
        points = (state.site_xpos[feet_ids] - state.xpos[calf_ids])

        osc_data = get_data_fn(model, state, points, body_ids)

        taskspace_targets = np.zeros((4, 6))
        kp = 100
        kd = 50
        targets = kp * (initial_feet_pos - state.site_xpos[feet_ids])
        taskspace_targets[:, :3] = targets

        prog_data = update_fn(taskspace_targets, osc_data)

        solution = solve_fn(prog_data, warmstart)

        dv, u = jnp.split(
            solution.params.primal[0],
            [osc_controller.dv_idx],
        )
        print(f'Iteration: {i}')
        print(f'Status: {solution.state.status}')
        print(f'Error: {solution.state.error}')
        print(f'Torques: {u}')

        warmstart = solution.params

        state = step_fn(model, state, u)
        state_history.append(state)

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

    sys = mjcf.load_model(mj_model)
    html_string = html.render(
        sys=sys,
        states=state_history,
        height="100vh",
        colab=False,
    )

    html_path = os.path.join(
        os.path.dirname(__file__),
        "visualization/visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)