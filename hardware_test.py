import functools
from absl import app
import os

import time

from unitree_bindings import unitree_api

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx
from brax.mjx import pipeline

from src.controllers.osc import hardware_utilities as hardware_utils
from src.controllers.osc import controller
from src.controllers.osc.controller import OSQPConfig

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

    # JIT Pipeline Functions:
    init_fn = jax.jit(pipeline.init)

    # Initialize OSC Controller:
    taskspace_targets = jnp.zeros((5, 6))
    osc_controller = controller.OSCController(
        model=mj_model,
        num_contacts=4,
        num_taskspace_targets=5,
        osqp_config=OSQPConfig(
            tol=1e-3,
            maxiter=4000,
            verbose=False,
            jit=True,
        ),
    )

    # JIT Controller Functions:
    get_data = functools.partial(hardware_utils.get_data, assume_contact=True)
    get_data_fn = jax.jit(get_data)
    update_fn = jax.jit(osc_controller.update)
    solve_fn = jax.jit(osc_controller.solve)

    initial_state = init_fn(model, q=q_init, qd=qd_init, ctrl=default_ctrl)

    # Initialize Unitree Driver:
    network_name = "eno2"
    unitree_driver = unitree_api.MotorController()
    unitree_driver.init(network_name)
    print("Unitree Driver Initialized.")
    time.sleep(1.0)

    # Send Default Command:
    cmd = unitree_api.MotorCommand()
    cmd.stiffness = [60.0] * 12
    cmd.damping = [5.0] * 12
    cmd.kp = [5.0] * 12
    cmd.kd = [2.0] * 12
    cmd.q_setpoint = [0.0, 0.9, -1.8] * 4
    unitree_driver.update_command(cmd)
    print("Running Default Command.")
    time.sleep(1.0)

    # Initialize Robot State:
    init_robot_state = hardware_utils.get_robot_state(unitree_driver)
    initial_data = hardware_utils.update_mj_data(model, initial_state, init_robot_state)

    # Initialize Values and Warmstart:
    body_points = jnp.expand_dims(initial_data.site_xpos[imu_id], axis=0)
    feet_points = initial_data.site_xpos[feet_ids]
    points = jnp.concatenate([body_points, feet_points])
    body_ids = jnp.concatenate([base_id, calf_ids])

    osc_data = get_data_fn(model, initial_data, init_robot_state, points, body_ids)

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

    num_iterations = 100
    print("Running Control Loop.")
    control_loop_start = time.time()
    timing = []
    for i in range(num_iterations):
        # Start timer after functions are compiled:
        start_time = time.time()
        
        # Get Body and Feet Points:
        body_points = np.expand_dims(data.site_xpos[imu_id], axis=0)
        feet_points = data.site_xpos[feet_ids]
        points = np.concatenate([body_points, feet_points])
        body_ids = np.concatenate([base_id, calf_ids])

        # Zero Acceleration Targets:
        taskspace_targets = np.zeros((5, 6))

        # Get Robot State:
        robot_state = hardware_utils.get_robot_state(unitree_driver)
        print("Body Angular Velocity: ", robot_state.body_velocity)

        # Get OSC Data:
        osc_data = get_data_fn(model, data, robot_state, points, body_ids)

        # Update QP:
        prog_data = update_fn(taskspace_targets, osc_data)

        # Solve QP:
        solution = solve_fn(prog_data, warmstart)
        dv, u, z = np.split(
            solution.params.primal[0],
            [osc_controller.dv_idx, osc_controller.u_idx],
        )
        warmstart = solution.params

        # Reorder Torque Command:
        torque_cmd = u[hardware_utils.convert_motor_idx]

        # Update Command:
        """
            Turning off position based feedback terms.
            Keep velocity feedback terms for damping.
        """
        cmd.stiffness = [0.0] * 12
        cmd.damping = [5.0] * 12
        cmd.kp = [0.0] * 12
        cmd.kd = [2.0] * 12
        cmd.q_setpoint = [0.0, 0.9, -1.8] * 4
        cmd.qd_setpoint = [0.0, 0.0, 0.0] * 4
        cmd.torque_feedforward = torque_cmd

        unitree_driver.update_command(cmd)

        # 50 Hz OSC Control Loop:
        execution_time = time.time() - start_time
        sleep_time = 0.02 - execution_time
        sleep_time = sleep_time if sleep_time > 0 else 0.0
        time.sleep(sleep_time)
        
        # Measure Control Rate Timing: 
        timing.append(time.time() - start_time)

    print(f"Average Iteration Time: {np.mean(timing)}")
    print(f"Median Iteration Time: {np.median(timing)}")

    print(f"Total Elapsed Time: {time.time() - control_loop_start}")
    print(f"Target Length: {num_iterations * 0.02} seconds.")

    # Stop Control Thread:
    unitree_driver.stop_control_thread()


if __name__ == '__main__':
    app.run(main)