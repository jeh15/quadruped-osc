from absl.testing import absltest

import jax
import numpy as np
import flax.linen as nn
import distrax

from src.module_types import identity_normalization_fn
from src.distribution_utilities import ParametricDistribution
from src.algorithms.ppo import network_utilities as ppo_networks

from src.training_utilities import policy_step, unroll_policy_steps
from brax.training.acting import actor_step, generate_unroll
from brax.envs import fast

jax.config.parse_flags_with_absl()


class TrainingUtilitiesTest(absltest.TestCase):
    def test_policy_step(self):
        rng_key = jax.random.key(seed=42)

        # Brax Environment:
        env = fast.Fast()
        state = env.reset(rng_key)

        # Network Params:
        layer_sizes = (32, 32)
        input_size = env.observation_size
        output_size = env.action_size
        normalization_params = None
        input_normalization_fn = identity_normalization_fn
        activation = nn.tanh
        kernel_initializer = nn.initializers.lecun_uniform()

        # Network:
        networks = ppo_networks.make_ppo_networks(
            observation_size=input_size,
            action_size=output_size,
            input_normalization_fn=input_normalization_fn,
            policy_layer_sizes=layer_sizes,
            value_layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_initializer,
            action_distribution=ParametricDistribution(
                distribution=distrax.Normal,
                bijector=distrax.Tanh(),
            ),
        )
        policy_generator = ppo_networks.make_inference_fn(networks)
        policy_params = networks.policy_network.init(rng_key)
        policy_fn = policy_generator([normalization_params, policy_params])

        # Refactored Policy Step:
        state, transition = policy_step(
            env=env,
            state=state,
            policy=policy_fn,
            key=rng_key,
        )

        # Brax Policy Step:
        brax_state, brax_transition = actor_step(
            env=env,
            env_state=state,
            policy=policy_fn,
            key=rng_key,
        )

        # Test State:
        np.testing.assert_array_almost_equal(
            state.obs, brax_state.obs,
        )

        # Test Transition Container:
        np.testing.assert_array_almost_equal(
            transition.observation, brax_transition.observation,
        )
        np.testing.assert_array_almost_equal(
            transition.action, brax_transition.action,
        )
        np.testing.assert_array_almost_equal(
            transition.reward, brax_transition.reward,
        )
        np.testing.assert_array_almost_equal(
            transition.termination, 1.0 - brax_transition.discount,
        )
        np.testing.assert_array_almost_equal(
            transition.next_observation, brax_transition.next_observation,
        )

    def test_unroll_policy_steps(self):
        rng_key = jax.random.key(seed=42)

        # Brax Environment:
        env = fast.Fast()
        state = env.reset(rng_key)

        # Network Params:
        layer_sizes = (32, 32)
        input_size = env.observation_size
        output_size = env.action_size
        normalization_params = None
        input_normalization_fn = identity_normalization_fn
        activation = nn.tanh
        kernel_initializer = nn.initializers.lecun_uniform()

        # Network:
        networks = ppo_networks.make_ppo_networks(
            observation_size=input_size,
            action_size=output_size,
            input_normalization_fn=input_normalization_fn,
            policy_layer_sizes=layer_sizes,
            value_layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_initializer,
            action_distribution=ParametricDistribution(
                distribution=distrax.Normal,
                bijector=distrax.Tanh(),
            ),
        )
        policy_generator = ppo_networks.make_inference_fn(networks)
        policy_params = networks.policy_network.init(rng_key)
        policy_fn = policy_generator([normalization_params, policy_params])

        # Refactored Unroll Policy Steps:
        final_state, transitions = unroll_policy_steps(
            env=env,
            state=state,
            policy=policy_fn,
            key=rng_key,
            num_steps=10,
        )

        # Brax Unroll Policy Steps:
        brax_final_state, brax_transitions = generate_unroll(
            env=env,
            env_state=state,
            policy=policy_fn,
            key=rng_key,
            unroll_length=10,
        )

        # Test State:
        np.testing.assert_array_almost_equal(
            final_state.obs, brax_final_state.obs,
        )

        # Test Transition Container:
        np.testing.assert_array_almost_equal(
            transitions.observation, brax_transitions.observation,
        )
        np.testing.assert_array_almost_equal(
            transitions.action, brax_transitions.action,
        )
        np.testing.assert_array_almost_equal(
            transitions.reward, brax_transitions.reward,
        )
        np.testing.assert_array_almost_equal(
            transitions.termination, 1.0 - brax_transitions.discount,
        )
        np.testing.assert_array_almost_equal(
            transitions.next_observation, brax_transitions.next_observation,
        )


if __name__ == '__main__':
    absltest.main()
