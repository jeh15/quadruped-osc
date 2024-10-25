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

# Types:
from jaxopt.base import KKTSolution
from brax.mjx.base import State
import brax

import matplotlib.pyplot as plt


jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_disable_jit', True)


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
            check_primal_dual_infeasability=False,
            tol=1e-3,
            maxiter=4000,
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

    start_time = time.time()
    # state_history = []
    # feet_targets = []
    # for i in range(1000):
    #     # Feet site position relative to calf body:
    #     points = (state.site_xpos[feet_ids] - state.xpos[calf_ids])

    #     # Feet site velocity:
    #     offset = brax.base.Transform.create(pos=points)
    #     foot_indices = calf_ids - 1
    #     foot_velocities = offset.vmap().do(state.xd.take(foot_indices)).vel

    #     # Get OSC Data:
    #     osc_data = get_data_fn(model, state, points, body_ids)

    #     # Calculate Taskspace Targets:
    #     kp = 500
    #     kd = 10
    #     magnitude, frequency = 0.01, model.opt.timestep
    #     feet_position_targets = jax.vmap(lambda x: jnp.array([
    #         magnitude * jnp.sin(frequency * i) + x[0],
    #         x[1],
    #         magnitude * jnp.cos(frequency * i) + (x[2] - magnitude),
    #     ]))(initial_feet_pos)
    #     feet_velocity_targets = jnp.array([
    #         magnitude * frequency * jnp.cos(frequency * i),
    #         0.0,
    #         -magnitude * frequency * jnp.sin(frequency * i),
    #     ] * 4).reshape(4, 3)
    #     feet_acceleration_targets = jnp.array([
    #         -magnitude * frequency**2 * jnp.sin(frequency * i),
    #         0.0,
    #         -magnitude * frequency**2 * jnp.cos(frequency * i),
    #     ] * 4).reshape(4, 3)
    #     targets = feet_acceleration_targets + kp * (
    #         feet_position_targets - state.site_xpos[feet_ids]
    #         ) + kd * (feet_velocity_targets - foot_velocities)
    #     taskspace_targets = jnp.concatenate(
    #         [targets, jnp.zeros((4, 3))], axis=-1,
    #     )

    #     # Update Program Data:
    #     prog_data = update_fn(taskspace_targets, osc_data)

    #     # Solve OSC Problem:
    #     solution = solve_fn(prog_data, warmstart)
    #     dv, u = jnp.split(
    #         solution.params.primal[0],
    #         [osc_controller.dv_idx],
    #     )
    #     warmstart = solution.params

    #     print(f'Iteration: {i}')
    #     print(f'Status: {solution.state.status}')

    #     # Step Forward:
    #     state = step_fn(model, state, u)
    #     state_history.append(state)
    #     feet_targets.append(feet_position_targets)

    # print(f"Python Loop Execution Time: {time.time() - start_time}")

    def loop(
        carry: Tuple[State, KKTSolution], xs: jax.Array,
    ) -> Tuple[Tuple[State, KKTSolution], Tuple[State, jax.Array]]:
        # Unpack Carry:
        state, warmstart = carry

        # Feet site position relative to calf body:
        points = (state.site_xpos[feet_ids] - state.xpos[calf_ids])

        # Feet site velocity:
        offset = brax.base.Transform.create(pos=points)
        foot_indices = calf_ids - 1
        foot_velocities = offset.vmap().do(state.xd.take(foot_indices)).vel

        # Get OSC Data:
        osc_data = get_data_fn(model, state, points, body_ids)

        # Calculate Taskspace Targets:
        taskspace_targets = jnp.zeros((4, 6))

        # Calculate Taskspace Targets:
        kp = 100
        kd = 10
        magnitude, frequency = 0.05, model.opt.timestep / 2.0
        feet_position_targets = jax.vmap(lambda x: jnp.array([
            magnitude * jnp.sin(frequency * xs) + x[0],
            x[1],
            -magnitude * jnp.cos(frequency * xs) + (x[2] + magnitude),
        ]))(initial_feet_pos)
        feet_velocity_targets = jnp.array([
            magnitude * frequency * jnp.cos(frequency * xs),
            0.0,
            magnitude * frequency * jnp.sin(frequency * xs),
        ] * 4).reshape(4, 3)
        feet_acceleration_targets = jnp.array([
            -magnitude * frequency**2 * jnp.sin(frequency * xs),
            0.0,
            magnitude * frequency**2 * jnp.cos(frequency * xs),
        ] * 4).reshape(4, 3)
        targets = feet_acceleration_targets + kp * (
            state.site_xpos[feet_ids] - feet_position_targets
            ) + kd * (foot_velocities -feet_velocity_targets)
        # taskspace_targets = jnp.concatenate(
        #     [targets, jnp.zeros((4, 3))], axis=-1,
        # )

        # Update Program Data:
        prog_data = update_fn(taskspace_targets, osc_data)

        # Solve OSC Problem:
        solution = solve_fn(prog_data, warmstart)
        dv, u = jnp.split(
            solution.params.primal[0],
            [osc_controller.dv_idx],
        )
        warmstart = solution.params

        # Step Forward:
        state = step_fn(model, state, u)

        return (state, warmstart), (state, feet_position_targets)

    # Run Loop:
    initial_state = init_fn(model, q=q_init, qd=qd_init, ctrl=default_ctrl)
    initial_warmstart = osc_controller.prog.init_params(
        init_x=init_x,
        params_obj=(prog_data.H, prog_data.f),
        params_eq=prog_data.A,
        params_ineq=(prog_data.lb, prog_data.ub),
    )
    iterations = jnp.arange(5000)
    start_time = time.time()
    (final_state, warmstart), (states, feet_targets) = jax.lax.scan(
        f=loop,
        init=(initial_state, initial_warmstart),
        xs=iterations,
    )
    print(f"JAX Loop Execution Time: {time.time() - start_time}")

    # Plot Feet Position vs Targets:
    fig, ax = plt.subplots(1, 1)
    ax.plot(states.site_xpos[:, feet_ids[0], 0], states.site_xpos[:, feet_ids[0], 2], label="Foot Trajectory", linewidth=2.0)
    ax.plot(feet_targets[:, 0, 0], feet_targets[:, 0, 2], label="Target", linestyle="--",  linewidth=2.0)
    ax.set_xlabel("X Position", fontsize=14)
    ax.set_ylabel("Z Position", fontsize=14)
    ax.axis("equal")
    ax.legend()
    plt.show()

    # # Plot Feet Position vs Targets:
    # feet_positions = []
    # foot_target = []
    # for i in range(len(state_history)):
    #     feet_positions.append(
    #         [state_history[i].site_xpos[feet_ids[0], 0], state_history[i].site_xpos[feet_ids[0], 2]]
    #     )
    #     foot_target.append([feet_targets[i][0, 0], feet_targets[i][0, 2]])

    # feet_positions = np.array(feet_positions)
    # foot_target = np.array(foot_target)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(feet_positions[:, 0], feet_positions[:, 1], label="Actual")
    # ax.plot(foot_target[:, 0], foot_target[:, 1], label="Target", linestyle="--")
    # ax.set_xlabel("X Position")
    # ax.set_ylabel("Z Position")
    # ax.axis("equal")
    # ax.legend()
    # plt.show()

    fig.savefig("feet_position_vs_targets.png")

    state_list = []
    num_steps = states.q.shape[0]
    for i in range(num_steps):
        state_list.append(
            State(
                q=states.q[i],
                qd=states.qd[i],
                x=brax.base.Transform(
                    pos=states.x.pos[i],
                    rot=states.x.rot[i],
                ),
                xd=brax.base.Motion(
                    vel=states.xd.vel[i],
                    ang=states.xd.ang[i],
                ),
                **data.__dict__,
            ),
        )

    sys = mjcf.load_model(mj_model)
    html_string = html.render(
        sys=sys,
        states=state_list,
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
