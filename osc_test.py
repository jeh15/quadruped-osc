from typing import Tuple
from absl import app
import os

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx
from brax.mjx import pipeline

from brax.io import mjcf, html

from src.controllers.osc import utilities as osc_utils
from src.controllers.osc import controller
from src.controllers.osc.controller import OSQPConfig

# Types:
from jaxopt.base import KKTSolution
from brax.mjx.base import State
import brax


jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_disable_jit', True)


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
    init_fn = jax.jit(pipeline.init)
    step_fn = jax.jit(pipeline.step)

    # Initialize OSC Controller:
    taskspace_targets = jnp.zeros((num_envs, 5, 6))
    osc_controller = controller.OSCController(
        model=mj_model,
        num_contacts=4,
        num_taskspace_targets=5,
        use_motor_model=False,
        osqp_config=OSQPConfig(
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

    # Initialize Values and Warmstart:
    num_steps = 5000
    kick_magnitude = 0.2
    kick_interval = 100
    base_position_target = state.site_xpos[imu_id]
    imu_offset = jnp.array([-0.02557, 0.0, 0.04232])
    body_points = jnp.expand_dims(state.site_xpos[imu_id], axis=0)
    feet_points = state.site_xpos[feet_ids]
    points = jnp.concatenate([body_points, feet_points])
    body_ids = jnp.concatenate([base_id, calf_ids])

    osc_data = get_data_fn(model, state, points, body_ids)

    prog_data = update_fn(taskspace_targets, osc_data)

    weight = jnp.linalg.norm(model.opt.gravity) * jnp.sum(model.body_mass)
    init_x = jnp.concatenate([
        jnp.zeros(osc_controller.dv_size),
        default_ctrl,
        jnp.array([0, 0, weight / 4] * 4),
    ])

    warmstart = osc_controller.prog.init_params(
        init_x=init_x,
        params_obj=(prog_data.H, prog_data.f),
        params_eq=prog_data.A,
        params_ineq=(prog_data.lb, prog_data.ub),
    )

    def loop(
        carry: Tuple[State, KKTSolution, jax.Array], xs: jax.Array,
    ) -> Tuple[Tuple[State, KKTSolution, jax.Array], State]:
        state, warmstart, key = carry
        key, subkey = jax.random.split(key)

        # Random Disturbance:
        kick_theta = jax.random.uniform(subkey, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(xs, kick_interval) == 0
        qvel = state.qvel
        qvel = qvel.at[:2].set(kick_magnitude * kick + qvel[:2])
        state = state.replace(qvel=qvel)

        # Get Body and Feet Points:
        body_points = jnp.expand_dims(data.site_xpos[imu_id], axis=0)
        feet_points = data.site_xpos[feet_ids]
        points = jnp.concatenate([body_points, feet_points])
        body_ids = jnp.concatenate([base_id, calf_ids])

        # Initial Position Target:
        kp = 100
        kd = 25
        imu_data = jnp.reshape(state.sensordata[-6:], (2, 3))
        base_target = kp * (base_position_target - imu_data[0]) + kd * (-imu_data[1])
        base_targets = jnp.concatenate(
            [jnp.expand_dims(base_target, axis=0), jnp.zeros((1, 3))], axis=-1,
        )
        feet_targets = jnp.zeros((4, 6))
        taskspace_targets = jnp.concatenate(
            [base_targets, feet_targets], axis=0,
        )

        # Foot Center Target:
        # kp = 100
        # kd = 25
        # imu_data = jnp.reshape(state.sensordata[-6:], (2, 3))
        # polygon_center = imu_offset - jnp.mean(feet_points, axis=0)
        # position_target = jnp.array([
        #     polygon_center[0], polygon_center[1], base_position_target[2],
        # ])
        # base_target = kp * (position_target - imu_data[0]) + kd * (-imu_data[1])
        # base_targets = jnp.concatenate(
        #     [jnp.expand_dims(base_target, axis=0), jnp.zeros((1, 3))], axis=-1,
        # )
        # feet_targets = jnp.zeros((4, 6))
        # taskspace_targets = jnp.concatenate(
        #     [base_targets, feet_targets], axis=0,
        # )

        # Zero Acceleration Targets:
        # taskspace_targets = jnp.zeros((5, 6))

        # Get OSC Data
        osc_data = get_data_fn(model, state, points, body_ids)

        # Update QP:
        prog_data = update_fn(taskspace_targets, osc_data)

        # Solve QP:
        solution = solve_fn(prog_data, warmstart)
        dv, u, z = jnp.split(
            solution.params.primal[0],
            [osc_controller.dv_idx, osc_controller.u_idx],
        )
        warmstart = solution.params

        # Step Simulation:
        state = step_fn(model, state, u)

        return (state, warmstart, subkey), state

    # Run Loop:
    initial_state = init_fn(model, q=q_init, qd=qd_init, ctrl=default_ctrl)
    key = jax.random.key(0)
    (final_state, _, _), states = jax.lax.scan(
        f=loop,
        init=(initial_state, warmstart, key),
        xs=jnp.arange(num_steps),
    )

    # Visualize:
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
