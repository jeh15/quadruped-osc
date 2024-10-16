from absl.testing import absltest

import jax
import numpy as np
import flax.linen as nn
import distrax

from src import networks
from brax.training import networks as brax_networks
from src.algorithms.ppo import network_utilities as ppo_networks
from brax.training.agents.ppo import networks as brax_ppo_networks
from src.module_types import identity_normalization_fn
from src.distribution_utilities import ParametricDistribution

jax.config.parse_flags_with_absl()


def rename_params(new_params, params):
    for i, key_name in enumerate(params['params'].keys()):
        new_key_name = f'hidden_{i}'
        new_params['params'][new_key_name] = params['params'][key_name]
    return new_params


class NetworksTest(absltest.TestCase):
    def test_networks(self):
        rng_key = jax.random.key(seed=42)

        # Network Params:
        layer_sizes = (32, 32)
        input_size = 64
        batch_size = 32
        output_size = 12
        normalization_params = None
        input_normalization_fn = identity_normalization_fn
        activation = nn.tanh
        kernel_initializer = nn.initializers.lecun_uniform()

        x = jax.random.normal(rng_key, (batch_size, input_size))

        # Refactored Network:
        refactored_policy_network = networks.make_policy_network(
            input_size=input_size,
            output_size=output_size,
            input_normalization_fn=input_normalization_fn,
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_initializer,
        )
        refactored_value_network = networks.make_value_network(
            input_size=input_size,
            input_normalization_fn=input_normalization_fn,
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_initializer,
        )

        # Brax Network:
        brax_policy_network = brax_networks.make_policy_network(
            param_size=output_size,
            obs_size=input_size,
            preprocess_observations_fn=input_normalization_fn,
            hidden_layer_sizes=layer_sizes,
            activation=activation,
        )
        brax_value_network = brax_networks.make_value_network(
            obs_size=input_size,
            preprocess_observations_fn=input_normalization_fn,
            hidden_layer_sizes=layer_sizes,
            activation=activation,
        )

        # Test:
        refactored_policy_params = refactored_policy_network.init(rng_key)
        refactored_value_params = refactored_value_network.init(rng_key)

        brax_policy_params = {'params': {}}
        brax_value_params = {'params': {}}

        # Kernel Initialization depends on key and layer names...
        # Rename the keys to match brax layer naming.
        brax_policy_params = rename_params(
            brax_policy_params, refactored_policy_params,
        )
        brax_value_params = rename_params(
            brax_value_params, refactored_value_params,
        )

        refactored_policy_output = refactored_policy_network.apply(
            normalization_params, refactored_policy_params, x,
        )
        refactored_value_output = refactored_value_network.apply(
            normalization_params, refactored_value_params, x,
        )

        brax_policy_output = brax_policy_network.apply(
            normalization_params, brax_policy_params, x,
        )
        brax_value_output = brax_value_network.apply(
            normalization_params, brax_value_params, x,
        )

        np.testing.assert_array_almost_equal(
            refactored_policy_output, brax_policy_output,
        )
        np.testing.assert_array_almost_equal(
            refactored_value_output, brax_value_output,
        )

    def test_ppo_networks(self):
        rng_key = jax.random.key(seed=42)

        # Network Params:
        layer_sizes = (32, 32)
        input_size = 64
        batch_size = 32
        output_size = 12
        normalization_params = None
        input_normalization_fn = identity_normalization_fn
        activation = nn.tanh
        kernel_initializer = nn.initializers.lecun_uniform()

        x = jax.random.normal(rng_key, (batch_size, input_size))

        # Refactored Network:
        networks = ppo_networks.make_ppo_networks(
            observation_size=input_size,
            action_size=output_size,
            input_normalization_fn=input_normalization_fn,
            policy_layer_sizes=layer_sizes,
            value_layer_sizes=layer_sizes,
            activation=activation,
            kernel_init=kernel_initializer,
        )
        policy_generator = ppo_networks.make_inference_fn(networks)

        brax_networks = brax_ppo_networks.make_ppo_networks(
            observation_size=input_size,
            action_size=output_size,
            preprocess_observations_fn=input_normalization_fn,
            policy_hidden_layer_sizes=layer_sizes,
            value_hidden_layer_sizes=layer_sizes,
            activation=activation,
        )
        brax_policy_generator = brax_ppo_networks.make_inference_fn(brax_networks)

        # Initalize Networks:
        policy_params = networks.policy_network.init(rng_key)
        value_params = networks.value_network.init(rng_key)

        brax_policy_params = {'params': {}}
        brax_value_params = {'params': {}}

        # Rename the keys to match brax layer naming.
        brax_policy_params = rename_params(
            brax_policy_params, policy_params,
        )
        brax_value_params = rename_params(
            brax_value_params, value_params,
        )

        # Test:
        policy_output = networks.policy_network.apply(
            normalization_params, policy_params, x,
        )
        value_output = networks.value_network.apply(
            normalization_params, value_params, x,
        )

        brax_policy_output = brax_networks.policy_network.apply(
            normalization_params, brax_policy_params, x,
        )
        brax_value_output = brax_networks.value_network.apply(
            normalization_params, brax_value_params, x,
        )

        policy_fn = policy_generator([normalization_params, policy_params])
        actions, policy_data = policy_fn(x, rng_key)

        brax_policy_fn = brax_policy_generator([normalization_params, brax_policy_params])
        brax_actions, brax_policy_data = brax_policy_fn(x, rng_key)

        np.testing.assert_array_almost_equal(
            policy_output, brax_policy_output,
        )
        np.testing.assert_array_almost_equal(
            value_output, brax_value_output,
        )
        np.testing.assert_array_almost_equal(
            actions, brax_actions,
        )
        np.testing.assert_array_almost_equal(
            policy_data['log_prob'], brax_policy_data['log_prob'],
        )
        np.testing.assert_array_almost_equal(
            policy_data['raw_action'], brax_policy_data['raw_action'],
        )


if __name__ == '__main__':
    absltest.main()
