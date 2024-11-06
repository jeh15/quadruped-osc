from typing import Any
from absl import app
import os

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

import flax.struct
import flax.serialization

from brax import base
from brax import envs
from brax import math
from brax.base import Motion, Transform, System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html

import mujoco

from src.controllers.osc import utilities as osc_utils
from src.controllers.osc import controller
from src.controllers.osc.controller import OSQPConfig


# Types:
PRNGKey = jax.Array
from jaxopt.base import KKTSolution


@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_linear_velocity: float = 1.5
    tracking_angular_velocity: float = 0.8
    # Penalties / Regularization Terms:
    orientation_regularization: float = -5.0
    linear_z_velocity: float = -2.0
    angular_xy_velocity: float = -0.05
    torque: float = -2e-4
    action_rate: float = -0.01
    stand_still: float = -0.5
    termination: float = -1.0
    foot_slip: float = -0.1
    # Gait Terms:
    air_time: float = 0.2
    target_air_time: float = 0.1
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.25
    kernel_alpha: float = 1.0


def domain_randomize(sys: System, rng: PRNGKey) -> tuple[System, System]:
    @jax.vmap
    def randomize_parameters(rng):
        key, subkey = jax.random.split(rng)
        # friction
        friction = jax.random.uniform(subkey, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)

        # actuator
        key, subkey = jax.random.split(subkey)
        gain_range = (-5, 5)
        param = jax.random.uniform(
            subkey, (1,), minval=gain_range[0], maxval=gain_range[1]
        ) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)

        return friction, gain, bias

    friction, gain, bias = randomize_parameters(rng)

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'actuator_gainprm': 0,
        'actuator_biasprm': 0,
    })

    sys = sys.tree_replace({
        'geom_friction': friction,
        'actuator_gainprm': gain,
        'actuator_biasprm': bias,
    })  # type: ignore

    return sys, in_axes


class UnitreeGo2Env(PipelineEnv):
    """Environment for training the Unitree Go1 quadruped joystick policy in MJX."""

    def __init__(
        self,
        filename: str = 'unitree_go2/scene_mjx.xml',
        config: RewardConfig = RewardConfig(),
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        **kwargs,
    ):
        filename = f'models/{filename}'
        self.filepath = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
            filename,
        )
        sys = mjcf.load(self.filepath)
        self.step_dt = 0.02
        sys = sys.tree_replace({'opt.timestep': 0.004})

        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
        )

        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        self.kernel_sigma = config.kernel_sigma
        self.kernel_alpha = config.kernel_alpha
        self.target_air_time = config.target_air_time
        config_dict = flax.serialization.to_state_dict(config)
        del config_dict['kernel_sigma']
        del config_dict['kernel_alpha']
        del config_dict['target_air_time']
        self.reward_config = config_dict

        self.base_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self.init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self.init_qd = jnp.zeros(sys.nv)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        self.joint_lb = jnp.array([
            -1.0472, -1.5708, -2.7227,
            -1.0472, -1.5708, -2.7227,
            -1.0472, -0.5236, -2.7227,
            -1.0472, -0.5236, -2.7227,
        ])
        self.joint_ub = jnp.array([
            1.0472, 3.4907, -0.83776,
            1.0472, 3.4907, -0.83776,
            1.0472, 4.5379, -0.83776,
            1.0472, 4.5379, -0.83776,
        ])
        # Torque limits:
        self.torque_limits = sys.mj_model.actuator_forcerange
        self.ctrl_lb = jnp.array(self.torque_limits[:, 0])
        self.ctrl_ub = jnp.array(self.torque_limits[:, 1])
        feet_site = [
            'front_left_foot',
            'front_right_foot',
            'hind_left_foot',
            'hind_right_foot',
        ]
        feet_site_idx = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_idx), 'Site not found.'
        self.feet_site_idx = np.array(feet_site_idx)
        calf_body = [
            'front_left_calf',
            'front_right_calf',
            'hind_left_calf',
            'hind_right_calf',
        ]
        calf_body_idx = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, c)
            for c in calf_body
        ]
        assert not any(id_ == -1 for id_ in calf_body_idx), 'Body not found.'
        self.calf_body_idx = np.array(calf_body_idx)
        base_body = [
            'base_link',
        ]
        base_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, b)
            for b in base_body
        ]
        self.base_body_idx = np.asarray(base_body_id)
        self.imu_site_idx = np.asarray(
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'imu'),
        )
        self.imu_offset = jnp.array([-0.02557, 0.0, 0.04232])
        self.foot_radius = 0.022
        self.history_length = 15
        self.num_observations = 31
        self.body_ids = np.concatenate([self.base_body_idx, self.calf_body_idx])

    def sample_command(self, rng: jax.Array) -> jax.Array:
        forward_velocity_range = [-0.6, 1.5]
        lateral_velocity_range = [-0.8, 0.8]
        yaw_rate_range = [-0.7, 0.7]

        _, forward_velocity_key, lateral_velocity_key, yaw_rate_key = jax.random.split(rng, 4)
        forward_velocity = jax.random.uniform(
            forward_velocity_key,
            (1,),
            minval=forward_velocity_range[0],
            maxval=forward_velocity_range[1],
        )
        lateral_velocity = jax.random.uniform(
            lateral_velocity_key,
            (1,),
            minval=lateral_velocity_range[0],
            maxval=lateral_velocity_range[1],
        )
        yaw_rate = jax.random.uniform(
            yaw_rate_key,
            (1,),
            minval=yaw_rate_range[0],
            maxval=yaw_rate_range[1],
        )
        new_cmd = jnp.array([
            forward_velocity[0], lateral_velocity[0], yaw_rate[0],
        ])
        return new_cmd

    def reset(self, rng: PRNGKey) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self.init_q, self.init_qd)

        # Caluclate World Points:
        body_points = jnp.expand_dims(
            pipeline_state.site_xpos[self.imu_site_idx], axis=0,
        )
        feet_points = pipeline_state.site_xpos[self.feet_site_idx]
        points = jnp.concatenate([body_points, feet_points])

        # Warmstart KKT Solution:
        taskspace_targets = jnp.zeros((self.num_taskspace_targets, 6))
        osc_data = osc_utils.get_data(self.sys.mj_model, pipeline_state, points, self.body_ids)
        prog_data = self.osc_controller.update(taskspace_targets, osc_data)

        weight = jnp.linalg.norm(self.sys.mj_model.opt.gravity) * jnp.sum(self.sys.mj_model.body_mass)
        init_x = jnp.concatenate([
            jnp.zeros(self.osc_controller.dv_size),
            jnp.zeros_like(self.default_ctrl),
            jnp.array([0, 0, weight / 4] * 4),
        ])
        warmstart = self.osc_controller.prog.init_params(
            init_x=init_x,
            params_obj=(prog_data.H, prog_data.f),
            params_eq=prog_data.A,
            params_ineq=(prog_data.lb, prog_data.ub),
        )

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(self.num_taskspace_targets * 6),
            'previous_velocity': jnp.zeros(12),
            'command': self.sample_command(key),
            'previous_contact': jnp.zeros(4, dtype=bool),
            'feet_air_time': jnp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'kick': jnp.array([0.0, 0.0]),
            'step': 0,
            'kkt_solution': warmstart,
            'previous_control': jnp.zeros(12),
        }

        observation_history = jnp.zeros(
            self.history_length * self.num_observations,
        )
        observation = self.get_observation(
            pipeline_state, state_info, observation_history,
        )

        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        metrics = {'total_distance': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info,
        )
        return state

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_key = jax.random.split(state.info['rng'], 3)

        # Distrubance:
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_key, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        # Format Action:
        reshaped_action = jnp.reshape(action, (self.num_taskspace_targets, 6))

        # Calculate OSC Control:
        pipeline_state = state.pipeline_state
        body_points = jnp.expand_dims(
            pipeline_state.site_xpos[self.imu_site_idx], axis=0,
        )
        feet_points = pipeline_state.site_xpos[self.feet_site_idx]
        points = jnp.concatenate([body_points, feet_points])

        osc_data = osc_utils.get_data(self.sys.mj_model, pipeline_state, points, self.body_ids)
        prog_data = self.osc_controller.update(reshaped_action, osc_data)

        # Solve QP:
        solution = self.osc_controller.solve(prog_data, state.info['kkt_solution'])
        dv, u, z = jnp.split(
            solution.params.primal[0],
            [self.osc_controller.dv_idx, self.osc_controller.u_idx],
        )
        warmstart = solution.params

        # Physics step:
        control_input = jnp.clip(u, self.ctrl_lb, self.ctrl_ub)
        pipeline_state = self.pipeline_step(
            state.pipeline_state, control_input,
        )
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        observation = self.get_observation(
            pipeline_state,
            state.info,
            state.obs,
        )
        joint_angles = pipeline_state.q[7:]
        joint_velocities = pipeline_state.qd[6:]

        # foot contact data based on z-position
        # pytype: disable=attribute-error
        foot_pos = pipeline_state.site_xpos[self.feet_site_idx]
        foot_contact_z = foot_pos[:, 2] - self.foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['previous_contact']
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info['previous_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        state.info['feet_air_time'] += self.dt

        # Done if joint limits are reached, robot is falling, or osc failed
        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[self.base_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.joint_lb)
        done |= jnp.any(joint_angles > self.joint_ub)
        done |= pipeline_state.x.pos[self.base_idx - 1, 2] < 0.15

        # Done if OSC failed:
        # done |= solution.state.status != 1

        # reward
        rewards = {
            'tracking_linear_velocity': (
                self._reward_tracking_velocity(state.info['command'], x, xd)
            ),
            'tracking_angular_velocity': (
                self._reward_tracking_yaw_rate(state.info['command'], x, xd)
            ),
            'linear_z_velocity': self._reward_vertical_velocity(xd),
            'angular_xy_velocity': self._reward_angular_velocity(xd),
            'orientation_regularization': self._reward_orientation_regularization(x),
            'torque': self._reward_torques(pipeline_state.qfrc_actuator[6:]),
            'action_rate': self._reward_action_rate(action, state.info['previous_action']),
            'stand_still': self._reward_stand_still(
                state.info['command'], joint_angles,
            ),
            'foot_slip': self._reward_foot_slip(
                pipeline_state, contact_filt_cm,
            ),
            'air_time': self._reward_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            'termination': jnp.float64(
                self._reward_termination(done, state.info['step'])
            ) if jax.config.x64_enabled else jnp.float32(
                self._reward_termination(done, state.info['step'])
            ),
        }
        rewards = {
            k: v * self.reward_config[k] for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # state management
        state.info['kick'] = kick
        state.info['previous_action'] = action
        state.info['previous_velocity'] = joint_velocities
        state.info['feet_air_time'] *= ~contact_filt_mm
        state.info['previous_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng
        state.info['kkt_solution'] = warmstart
        state.info['previous_control'] = control_input

        # sample new command if more than 500 timesteps achieved
        state.info['command'] = jnp.where(
            state.info['step'] > 500,
            self.sample_command(cmd_rng),
            state.info['command'],
        )
        # reset the step counter when done
        state.info['step'] = jnp.where(
            done | (state.info['step'] > 500), 0, state.info['step']
        )

        # Proxy Metrics:
        state.metrics['total_distance'] = math.normalize(
            x.pos[self.base_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        state = state.replace(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
        )
        return state

    def get_observation(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        observation_history: jax.Array,
    ) -> jax.Array:
        """
            Observation: [
                yaw_rate,
                projected_gravity,
                command,
                relative_motor_positions,
                previous_action,
            ]

            or test using OSC control

            Observation: [
                yaw_rate,
                projected_gravity,
                command,
                relative_motor_positions,
                previous_control,
            ]

        """
        inverse_trunk_rotation = math.quat_inv(pipeline_state.x.rot[0])
        body_frame_yaw_rate = math.rotate(
            pipeline_state.xd.ang[0], inverse_trunk_rotation,
        )[2]
        projected_gravity = math.rotate(
            jnp.array([0, 0, -1]), inverse_trunk_rotation,
        )

        q = pipeline_state.q[7:]

        observation = jnp.concatenate([
            jnp.array([body_frame_yaw_rate]),
            projected_gravity,
            state_info['command'],
            q - self.default_pose,
            state_info['previous_action'],
        ])

        # clip, noise
        observation = (
            jnp.clip(observation, -100.0, 100.0)
            + self._obs_noise
            * jax.random.uniform(
                state_info['rng'],
                observation.shape,
                minval=-1,
                maxval=1,
            )
        )
        # stack observations through time
        observation = jnp.roll(
            observation_history, observation.size
        ).at[:observation.size].set(observation)

        return observation

    def _reward_vertical_velocity(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2])

    def _reward_angular_velocity(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(xd.ang[0, :2]))

    def _reward_orientation_regularization(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jnp.array([0.0, 0.0, 1.0])
        deviation = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(deviation[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(
        self, action: jax.Array, previous_action: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(action - previous_action))

    def _reward_tracking_velocity(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        base_velocity = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        error = jnp.sum(jnp.square(commands[:2] - base_velocity[:2]))
        return jnp.exp(-error / self.kernel_sigma)

    def _reward_tracking_yaw_rate(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_yaw_rate = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        error = jnp.square(commands[2] - base_yaw_rate[2])
        return jnp.exp(-error / self.kernel_sigma)

    def _reward_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Flight Phase Reward:
        reward_air_time = jnp.sum((air_time - self.target_air_time) * first_contact)
        reward_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return reward_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jnp.sum(jnp.abs(joint_angles - self.default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, pipeline_state: base.State, contact_filter: jax.Array
    ) -> jax.Array:
        # Foot Velocity:
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self.feet_site_idx]
        feet_offset = pos - pipeline_state.xpos[self.calf_body_idx]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self.calf_body_idx - 1
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(foot_vel[:, :2]) * contact_filter.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

    def taskspace_points(self, pipeline_state: State) -> jax.Array:
        """ Returns world points of taskspace target bodies. """
        # (num_envs, body_id, 3)
        body_points = jnp.expand_dims(
            pipeline_state.site_xpos[:, self.imu_site_idx], axis=1,
        )
        feet_points = pipeline_state.site_xpos[:, self.feet_site_idx]
        points = jnp.concatenate([body_points, feet_points], axis=1)
        return points

    def np_observation(
        self,
        mj_data: mujoco.MjData,
        command: np.ndarray,
        previous_action: np.ndarray,
        observation_history: np.ndarray,
    ) -> np.ndarray:
        # Numpy implementation of the observation function:
        def rotate(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
            if len(vec.shape) != 1:
                raise ValueError('vec must have no batch dimensions.')
            s, u = quat[0], quat[1:]
            r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
            r = r + 2 * s * np.cross(u, vec)
            return r

        def quat_inv(q: np.ndarray) -> np.ndarray:
            return q * np.array([1, -1, -1, -1])

        base_w = mj_data.qpos[3:7]
        base_dw = mj_data.qvel[3:6]
        q = mj_data.qpos[7:]

        inverse_trunk_rotation = quat_inv(base_w)
        body_frame_yaw_rate = rotate(
            base_dw, inverse_trunk_rotation,
        )[2]
        projected_gravity = rotate(
            jnp.array([0, 0, -1]), inverse_trunk_rotation,
        )

        new_observation = np.concatenate([
            np.array([body_frame_yaw_rate]),
            projected_gravity,
            command,
            q - self.default_ctrl,
            previous_action,
        ])

        # clip, noise
        new_observation = (
            jnp.clip(new_observation, -100.0, 100.0)
            + self._obs_noise * np.random.uniform(
                low=-1, high=1, size=new_observation.shape,
            )
        )
        # stack observations through time
        observation = np.roll(observation_history, new_observation.size)
        observation[:new_observation.size] = new_observation

        return observation


envs.register_environment('unitree_go2', UnitreeGo2Env)


def main(argv=None):
    env = UnitreeGo2Env(filename='unitree_go2/scene_barkour_hfield_mjx.xml')
    rng = jax.random.PRNGKey(0)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    state = reset_fn(rng)

    num_steps = 100
    states = []
    for i in range(num_steps):
        print(f"Step: {i}")
        state = step_fn(state, jnp.zeros_like(env.default_ctrl))
        states.append(state.pipeline_state)

    html_string = html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
        states=states,
        height="100vh",
        colab=False,
    )
    html_path = os.path.join(
        os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        "visualization/visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
