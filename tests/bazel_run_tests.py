import unitree_api

import os
import pickle
import copy

import numpy as np

import time


def generate_random_control(trajectory_length: int = 100) -> list[float]:
    defualt_ctrl = np.asarray([0.0, 0.9, -1.8] * 4)
    amplitude = np.random.uniform(low=-1.5, high=1.5)
    frequency = np.random.uniform(low=-0.01, high=0.01)
    ctrl = defualt_ctrl[1] + amplitude * np.sin(frequency * np.arange(trajectory_length))
    ctrl_trajectory = np.tile(defualt_ctrl, (trajectory_length, 1))
    ctrl_trajectory[:, 1] = ctrl
    
    # Saturate Control Input:
    lb = np.asarray([
        -1.0472, -1.5708, -2.7227,
        -1.0472, -1.5708, -2.7227,
        -1.0472, -0.5236, -2.7227,
        -1.0472, -0.5236, -2.7227,
    ])
    ub = np.asarray([
        1.0472, 3.4907, -0.83776,
        1.0472, 3.4907, -0.83776,
        1.0472, 4.5379, -0.83776,
        1.0472, 4.5379, -0.83776,
    ])
    ctrl_trajectory = np.fromiter(
        iter=map(lambda x: np.clip(x, lb, ub), ctrl_trajectory),
        dtype=np.dtype((float, 12)),
    )
    return ctrl_trajectory.tolist()


def reset_position(current_postion: list[float], trajectory_length: int = 100) -> list[float]:
    defualt_ctrl = np.asarray([0.0, 0.9, -1.8] * 4)
    ctrl_trajectory = np.linspace(current_postion, defualt_ctrl, trajectory_length, axis=0)

    # Saturate Control Input:
    lb = np.asarray([
        -1.0472, -1.5708, -2.7227,
        -1.0472, -1.5708, -2.7227,
        -1.0472, -0.5236, -2.7227,
        -1.0472, -0.5236, -2.7227,
    ])
    ub = np.asarray([
        1.0472, 3.4907, -0.83776,
        1.0472, 3.4907, -0.83776,
        1.0472, 4.5379, -0.83776,
        1.0472, 4.5379, -0.83776,
    ])
    ctrl_trajectory = np.fromiter(
        iter=map(lambda x: np.clip(x, lb, ub), ctrl_trajectory),
        dtype=np.dtype((float, 12)),
    )

    return ctrl_trajectory.tolist()



def main(argv=None):
    # Test Bindings by reading Robot Values:
    network_name = "eno2"
    unitree = unitree_api.MotorController()
    unitree.init(network_name)

    time.sleep(1.0)

    # Get Low State:
    low_state = unitree.get_low_state()
    print(f"Low State - Foot Force: {low_state.foot_force}")

    # Get IMU State:
    imu_state = unitree.get_imu_state()
    print(f"IMU State - Quaternion: {imu_state.quaternion}")
    print(f"IMU State - Gyroscope: {imu_state.gyroscope}")
    print(f"IMU State - Accelerometer: {imu_state.accelerometer}")
    print(f"IMU State - RPY: {imu_state.rpy}")

    # Get Motor State:
    motor_state = unitree.get_motor_state()
    print(f"Motor State - Motor Angles: {motor_state.q}")
    print(f"Motor State - Motor Velocities: {motor_state.qd}")
    print(f"Motor State - Motor Accelertations: {motor_state.qdd}")
    print(f"Motor State - Motor Torques: {motor_state.torque_estimate}")

    # Try some manual motor control:
    cmd = unitree_api.MotorCommand()
    cmd.stiffness = [60.0] * 12
    cmd.damping = [5.0] * 12
    cmd.kp = [5.0] * 12
    cmd.kd = [2.0] * 12
    cmd.q_setpoint = [0.0, 0.9, -1.8] * 4
    unitree.update_command(cmd)
    time.sleep(1.0)

    # Right Leg Sinewave:
    num_runs = 10
    trajectory_length = 250
    return_length = 100
    run_history = []
    for run in range(num_runs):
        # Generate trajectory and reset to default position:
        motor_states = []
        control_trajectories = generate_random_control(trajectory_length)
        cmd.q_setpoint = [0.0, 0.9, -1.8] * 4
        unitree.update_command(cmd)
        time.sleep(1.0)
        for ctrl in control_trajectories:
            cmd.q_setpoint = ctrl
            unitree.update_command(cmd)
            motor_states.append(copy.deepcopy(unitree.get_motor_state()))
            time.sleep(0.02)
        run_history.append(motor_states)

        print(f"Run: {run} - Resetting to Default Position")
        default_trajectory = reset_position(motor_states[-1].q, return_length)
        for ctrl in default_trajectory:
            cmd.q_setpoint = ctrl
            unitree.update_command(cmd)
            time.sleep(0.02)
    
    # Save Motor States:
    with open('motor_states.pkl', "wb") as f:
        pickle.dump(run_history, f)


if __name__ == '__main__':
    main()
