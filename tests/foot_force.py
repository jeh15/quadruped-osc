from unitree_bindings import unitree_api

import os
import pickle
import copy
import functools

import jax
import jax.numpy as jnp

import numpy as np

import time

PRNGKey = jax.typing.ArrayLike


def main(argv=None):
    # Test Bindings by reading Robot Values:
    network_name = "eno2"
    unitree = unitree_api.MotorController()
    unitree.init(network_name)

    time.sleep(1.0)

    # Get Low State:
    while(True):
        try:
            low_state = unitree.get_low_state()
            print(f"Low State - Foot Force: {low_state.foot_force}")
        except KeyboardInterrupt:
            break

    # Stop Control Thread:
    unitree.stop_control_thread()


if __name__ == '__main__':
    main()
