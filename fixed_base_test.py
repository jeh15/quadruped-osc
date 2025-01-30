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
import brax
from brax.mjx import pipeline

from src.controllers.osc import hardware_utilities as hardware_utils
from src.controllers.osc.tests._fixed_base_controller import OSCController, OSQPConfig

# Debug:
from brax.io import mjcf, html

jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_disable_jit', True)


def main(argv):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'models/unitree_go2/go2_mjx_fixed.xml',
    )
    # Mujoco model:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

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

    # JIT Pipeline Functions:
    init_fn = jax.jit(pipeline.init)

    # Used to visualize for debugging:
    step_fn = jax.jit(pipeline.step)

    # Separate functions:
    update_mj_data = functools.partial(hardware_utils.update_mj_data, fixed_base=True)
    update_mj_data_fn = jax.jit(update_mj_data)
    get_osc_data = functools.partial(hardware_utils.get_osc_data, assume_contact=False)
    get_osc_data_fn = jax.jit(get_osc_data)

    # Initialize Unitree Driver:
    network_name = "eno2"
    unitree_driver = unitree_api.MotorController()
    unitree_driver.init(network_name)
    print("Unitree Driver Initialized.")
    time.sleep(1.0)

    # Go to Default Position:
    print("Moving to Default Position.")
    hardware_utils.default_position(unitree_driver, trajectory_length=200)
    time.sleep(1.0)

    # Initialize Default Command and Hold Position:
    cmd = unitree_api.MotorCommand()
    cmd.stiffness = [60.0] * 12
    cmd.damping = [5.0] * 12
    cmd.kp = [5.0] * 12
    cmd.kd = [2.0] * 12
    cmd.q_setpoint = [0.0, 0.9, -1.8] * 4
    unitree_driver.update_command(cmd)
    print("Running Default Command and Holding Position.")
    time.sleep(1.0)

    # Initialize mj_data:
    initial_robot_state = hardware_utils.get_robot_state(unitree_driver)
    q_initial = jnp.asarray(initial_robot_state.motor_position)
    qd_initial = jnp.asarray(initial_robot_state.motor_velocity)
    default_ctrl = jnp.zeros(12)

    data = data.replace(
        qpos=q_initial, qvel=qd_initial, ctrl=default_ctrl,
    )
    data = mjx.forward(model, data)

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

    update_fn = jax.jit(osc_controller.update)
    solve_fn = jax.jit(osc_controller.solve)

    # Record Initial Feet Positions:
    initial_feet_positions = data.site_xpos[feet_ids]

    # Initialize Values and Warmstart:
    points = initial_feet_positions
    body_ids = calf_ids

    osc_data = get_osc_data_fn(model, data, initial_robot_state, points, body_ids)

    prog_data = update_fn(taskspace_targets, osc_data)

    init_x = jnp.concatenate([
        jnp.zeros(osc_controller.dv_size),
        default_ctrl,
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

    # Debugging:
    debug_states = []
    robot_state = hardware_utils.get_robot_state(unitree_driver)
    q = jnp.asarray(robot_state.motor_position)
    qd = jnp.asarray(robot_state.motor_velocity)
    state = init_fn(model, q=q, qd=qd, ctrl=jnp.zeros_like(default_ctrl))

    for i in range(num_iterations):
        # Start timer after functions are compiled:
        start_time = time.time()

        # Get Robot State:
        robot_state = hardware_utils.get_robot_state(unitree_driver)

        print(f"Robot Motor Positions: {robot_state.motor_position}")
        print(f"Robot Motor Velocities: {robot_state.motor_velocity}")

        # Update mj_data:
        data = update_mj_data_fn(model, data, robot_state)
        
        # Feet site position relative to calf body:
        points = data.site_xpos[feet_ids]

        # Feet site velocity: (This is correct compared to mujoco framelinvel sensor)
        def feet_velocity_fn(data):
            state = init_fn(model, q=data.qpos, qd=data.qvel, ctrl=data.ctrl)
            relative_position = state.site_xpos[feet_ids] - state.xpos[calf_ids]
            offset = brax.base.Transform.create(pos=relative_position)
            foot_indices = calf_ids - 1
            return offset.vmap().do(state.xd.take(foot_indices)).vel
        
        foot_velocities = jax.jit(feet_velocity_fn)(data)

        # Zero Acceleration Targets:
        taskspace_targets = np.zeros((4, 6))

        # Calculate Taskspace Targets: Circle
        # kp = 1000
        # kd = 10
        # magnitude, frequency = 0.05, model.opt.timestep
        # feet_position_targets = jax.vmap(lambda x: jnp.array([
        #     magnitude * jnp.sin(frequency * i) + x[0],
        #     x[1],
        #     -magnitude * jnp.cos(frequency * i) + (x[2] + magnitude),
        # ]))(initial_feet_positions)
        # feet_velocity_targets = jnp.array([
        #     magnitude * frequency * jnp.cos(frequency * i),
        #     0.0,
        #     magnitude * frequency * jnp.sin(frequency * i),
        # ] * 4).reshape(4, 3)
        # feet_acceleration_targets = jnp.array([
        #     -magnitude * frequency**2 * jnp.sin(frequency * i),
        #     0.0,
        #     magnitude * frequency**2 * jnp.cos(frequency * i),
        # ] * 4).reshape(4, 3)
        # targets = feet_acceleration_targets + kp * (
        #     feet_position_targets - data.site_xpos[feet_ids]
        #     ) + kd * (feet_velocity_targets - foot_velocities)
        # taskspace_targets = jnp.concatenate(
        #     [targets, jnp.zeros((4, 3))], axis=-1,
        # )

        # Get OSC Data:
        osc_data = get_osc_data_fn(model, data, robot_state, points, body_ids)

        # Update QP:
        prog_data = update_fn(taskspace_targets, osc_data)

        # Solve QP:
        solution = solve_fn(prog_data, warmstart)
        dv, u = np.split(
            solution.params.primal[0],
            [osc_controller.dv_idx],
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

        # Command Torque:
        unitree_driver.update_command(cmd)

        # Generate Simulation View: (Debugging)
        # state = step_fn(model, state, u)
        # debug_states.append(state)

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

    # # Debugging:
    # sys = mjcf.load_model(mj_model)
    # html_string = html.render(
    #     sys=sys,
    #     states=debug_states,
    #     height="100vh",
    #     colab=False,
    # )

    # html_path = os.path.join(
    #     os.path.dirname(__file__),
    #     "visualization/hardware_visualization.html",
    # )

    # with open(html_path, "w") as f:
    #     f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)