from unitree_bindings import unitree_api

import os
import pickle

import numpy as np

import pdb


def main(argv=None):
    # Load data
    ws = os.path.dirname(__file__)
    data_filepath = os.path.join(ws, 'data/motor_states.pkl')
    with open(data_filepath, 'rb') as f:
        motor_state_data = pickle.load(f)

    # Sort data:
    data = {}
    for run_iteration, run_data in enumerate(motor_state_data):
        data[f'run_{run_iteration}'] = {}
        position = []
        velocity = []
        acceleration = []
        torque = []

        # Extract motor states:
        for motor_state in run_data:
            position.append(motor_state.q)
            velocity.append(motor_state.qd)
            acceleration.append(motor_state.qdd)
            torque.append(motor_state.torque_estimate)

        # Convert to numpy arrays:
        position = np.asarray(position)
        velocity = np.asarray(velocity)
        acceleration = np.asarray(acceleration)
        torque = np.asarray(torque)

        # Save to dictionary:
        data[f'run_{run_iteration}']['position'] = position
        data[f'run_{run_iteration}']['velocity'] = velocity
        data[f'run_{run_iteration}']['acceleration'] = acceleration
        data[f'run_{run_iteration}']['torque'] = torque


    # Save data:
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)



if __name__ == '__main__':
    main()
