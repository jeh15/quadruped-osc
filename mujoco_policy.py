from absl import app, flags
import functools
import time

import pygame

import jax
import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer

import matplotlib.pyplot as plt

from src.envs import unitree_go2
from src.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)
pygame.init()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)

def controller(
    action: npt.ArrayLike,
    default_control: npt.ArrayLike,
    ctrl_lb: npt.ArrayLike,
    ctrl_ub: npt.ArrayLike,
    action_scale: float,
) -> np.ndarray:
    motor_targets = default_control + action * action_scale
    motor_targets = np.clip(motor_targets, ctrl_lb, ctrl_ub)
    return motor_targets


def rotate(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    if len(vec.shape) != 1:
        raise ValueError('vec must have no batch dimensions.')
    s, u = quat[0], quat[1:]
    r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
    r = r + 2 * s * np.cross(u, vec)
    return r


def quat_inv(q: np.ndarray) -> np.ndarray:
    return q * np.array([1, -1, -1, -1])


def main(argv=None):
    # Load from Env:
    env = unitree_go2.UnitreeGo2Env(filename='unitree_go2/scene_barkour_hfield_mjx.xml')
    model = env.sys.mj_model

    data = mujoco.MjData(model)  # type: ignore
    mujoco.mj_resetData(model, data)  # type: ignore
    control_rate = 0.02
    num_steps = int(control_rate / model.opt.timestep)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Controller:
    controller_fn = functools.partial(
        controller,
        default_control=env.default_ctrl,
        action_scale=env._action_scale,
        ctrl_lb=env.ctrl_lb,
        ctrl_ub=env.ctrl_ub,
    )

    # Test:
    data.qpos = model.key_qpos.flatten()
    command = np.array([1.0, 0.0, 0.0])
    action = model.key_ctrl.flatten()
    observation = np.zeros(env.history_length * env.num_observations)

    # Setup Joystick:
    joysticks = {}

    key = jax.random.key(0)
    termination_flag = False
    command_history = []

    global_steps = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 5 

        while viewer.is_running() and not termination_flag:
            if global_steps >= 1000:
                termination_flag = True

            step_time = time.time()
            action_rng, key = jax.random.split(key)
            observation = env.np_observation(
                mj_data=data,
                command=command,
                previous_action=action,
                observation_history=observation,
            )
            action, _ = inference_fn(observation, action_rng)
            ctrl = controller_fn(action)
            data.ctrl = ctrl

            for _ in range(num_steps):
                mujoco.mj_step(model, data)  # type: ignore

            viewer.sync()

            sleep_time = control_rate - (time.time() - step_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            global_steps += 1


if __name__ == '__main__':
    app.run(main)
