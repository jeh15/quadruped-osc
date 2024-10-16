from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from brax.envs import fast
from brax.envs.wrappers.training import wrap
from src.algorithms.ppo.network_utilities import PPONetworkParams
import src.algorithms.ppo.network_utilities as ppo_networks
from src.module_types import identity_normalization_fn
from src.training_utilities import unroll_policy_steps
from brax.training.agents.ppo import networks as brax_ppo_networks
from brax.training.agents.ppo.losses import PPONetworkParams as BraxPPONetworkParams
from brax.training.acting import generate_unroll

# Test Case Import:
from src.algorithms.ppo.loss_utilities import calculate_gae, loss_function
from brax.training.agents.ppo.losses import compute_gae, compute_ppo_loss

jax.config.parse_flags_with_absl()


def rename_params(new_params, params):
    for i, key_name in enumerate(params['params'].keys()):
        new_key_name = f'hidden_{i}'
        new_params['params'][new_key_name] = params['params'][key_name]
    return new_params


class LossUtilitiesTest(absltest.TestCase):
    def test_gae_fn(self):
        rng_key = jax.random.key(seed=42)
        shape = (10, 3)
        bootstrap_shape = (3,)
        rewards = jax.random.normal(
            rng_key, shape=shape,
        )
        values = jax.random.normal(
            rng_key, shape=shape,
        )
        bootstrap_value = jax.random.normal(
            rng_key, shape=bootstrap_shape,
        )
        truncation_mask = jax.random.randint(
            rng_key, shape=shape, minval=0, maxval=2,
        )
        termination_mask = jax.random.randint(
            rng_key, shape=shape, minval=0, maxval=2,
        )

        # Refactored Function:
        returns, advantages = calculate_gae(
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            truncation_mask=truncation_mask,
            termination_mask=termination_mask,
        )

        # Brax Function:
        brax_returns, brax_advantages = compute_gae(
            truncation=1 - truncation_mask,
            termination=1 - termination_mask,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            lambda_=0.95,
            discount=0.99,
        )

        # Tests:
        np.testing.assert_array_almost_equal(returns, brax_returns)
        np.testing.assert_array_almost_equal(advantages, brax_advantages)

    def test_loss_function(self):
        rng_key = jax.random.key(seed=42)

        # Brax Environment:
        env = fast.Fast()
        env = wrap(
            env,
            episode_length=10,
            action_repeat=1,
        )

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
        )
        policy_params = networks.policy_network.init(rng_key)
        value_params = networks.value_network.init(rng_key)
        policy_generator = ppo_networks.make_inference_fn(networks)
        policy_fn = policy_generator([normalization_params, policy_params])

        brax_networks = brax_ppo_networks.make_ppo_networks(
            observation_size=input_size,
            action_size=output_size,
            preprocess_observations_fn=input_normalization_fn,
            policy_hidden_layer_sizes=layer_sizes,
            value_hidden_layer_sizes=layer_sizes,
            activation=activation,
        )
        brax_policy_params = {'params': {}}
        brax_value_params = {'params': {}}

        # Rename the keys to match brax layer naming.
        brax_policy_params = rename_params(
            brax_policy_params, policy_params,
        )
        brax_value_params = rename_params(
            brax_value_params, value_params,
        )

        # Unroll Policy Steps:
        num_steps = 10
        batch_size = 32
        num_minibatches = 4
        num_envs = 1

        process_count = jax.process_count()
        local_device_count = jax.local_device_count()
        local_devices_to_use = local_device_count

        key_envs = jax.random.split(rng_key, num_envs // process_count)
        key_envs = jnp.reshape(
            key_envs,
            (local_devices_to_use, -1) + key_envs.shape[1:],
        )
        state = env.reset(key_envs)

        # Needs batch dimension:
        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = unroll_policy_steps(
                env,
                current_state,
                policy_fn,
                current_key,
                num_steps,
                extra_fields=('truncation',)
            )
            return (next_state, next_key), data

        _, transitions = jax.lax.scan(
            f,
            (state, rng_key),
            None,
            length=batch_size * num_minibatches // num_envs,
        )

        def brax_f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = generate_unroll(
                env,
                current_state,
                policy_fn,
                current_key,
                num_steps,
                extra_fields=('truncation',)
            )
            return (next_state, next_key), data

        _, brax_transitions = jax.lax.scan(
            brax_f,
            (state, rng_key),
            None,
            length=batch_size * num_minibatches // num_envs,
        )

        params = PPONetworkParams(
            policy_params=policy_params,
            value_params=value_params,
        )

        brax_params = BraxPPONetworkParams(
            policy=brax_policy_params,
            value=brax_value_params,
        )

        # Loss Function:
        loss, metrics = loss_function(
            params=params,
            ppo_networks=networks,
            normalization_params=normalization_params,
            data=transitions,
            rng_key=rng_key,
            clip_coef=0.2,
            value_coef=0.25,
            entropy_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantages=False,
        )

        # Brax Loss Function:
        brax_loss, brax_metrics = compute_ppo_loss(
            params=brax_params,
            normalizer_params=normalization_params,
            data=brax_transitions,
            rng=rng_key,
            ppo_network=brax_networks,
            entropy_cost=0.01,
            discounting=0.99,
            gae_lambda=0.95,
            clipping_epsilon=0.2,
            normalize_advantage=False,
        )

        np.testing.assert_array_almost_equal(loss, brax_loss)

    def test_loss_gradient(self):
        rng_key = jax.random.key(seed=42)

        # Brax Environment:
        env = fast.Fast()
        env = wrap(
            env,
            episode_length=10,
            action_repeat=1,
        )

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
        )
        policy_params = networks.policy_network.init(rng_key)
        value_params = networks.value_network.init(rng_key)
        policy_generator = ppo_networks.make_inference_fn(networks)
        policy_fn = policy_generator([normalization_params, policy_params])

        brax_networks = brax_ppo_networks.make_ppo_networks(
            observation_size=input_size,
            action_size=output_size,
            preprocess_observations_fn=input_normalization_fn,
            policy_hidden_layer_sizes=layer_sizes,
            value_hidden_layer_sizes=layer_sizes,
            activation=activation,
        )
        brax_policy_params = {'params': {}}
        brax_value_params = {'params': {}}

        # Rename the keys to match brax layer naming.
        brax_policy_params = rename_params(
            brax_policy_params, policy_params,
        )
        brax_value_params = rename_params(
            brax_value_params, value_params,
        )

        # Unroll Policy Steps:
        num_steps = 10
        batch_size = 32
        num_minibatches = 4
        num_envs = 1

        process_count = jax.process_count()
        local_device_count = jax.local_device_count()
        local_devices_to_use = local_device_count

        key_envs = jax.random.split(rng_key, num_envs // process_count)
        key_envs = jnp.reshape(
            key_envs,
            (local_devices_to_use, -1) + key_envs.shape[1:],
        )
        state = env.reset(key_envs)

        # Needs batch dimension:
        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = unroll_policy_steps(
                env,
                current_state,
                policy_fn,
                current_key,
                num_steps,
                extra_fields=('truncation',)
            )
            return (next_state, next_key), data

        _, transitions = jax.lax.scan(
            f,
            (state, rng_key),
            None,
            length=batch_size * num_minibatches // num_envs,
        )

        def brax_f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = generate_unroll(
                env,
                current_state,
                policy_fn,
                current_key,
                num_steps,
                extra_fields=('truncation',)
            )
            return (next_state, next_key), data

        _, brax_transitions = jax.lax.scan(
            brax_f,
            (state, rng_key),
            None,
            length=batch_size * num_minibatches // num_envs,
        )

        params = PPONetworkParams(
            policy_params=policy_params,
            value_params=value_params,
        )

        brax_params = BraxPPONetworkParams(
            policy=brax_policy_params,
            value=brax_value_params,
        )

        loss_fn = lambda params: loss_function(
            params=params,
            ppo_networks=networks,
            normalization_params=normalization_params,
            data=transitions,
            rng_key=rng_key,
            clip_coef=0.2,
            value_coef=0.25,
            entropy_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantages=False,
        )
        brax_loss_fn = lambda params: compute_ppo_loss(
            params=params,
            normalizer_params=normalization_params,
            data=brax_transitions,
            rng=rng_key,
            ppo_network=brax_networks,
            entropy_cost=0.01,
            discounting=0.99,
            gae_lambda=0.95,
            clipping_epsilon=0.2,
            normalize_advantage=False,
        )

        loss_gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
        brax_loss_gradient_fn = jax.value_and_grad(brax_loss_fn, has_aux=True)

        # Loss Function:
        (loss, metrics), grad = loss_gradient_fn(
            params,
        )

        # Brax Loss Function:
        (brax_loss, brax_metrics), brax_grad = brax_loss_gradient_fn(
            brax_params,
        )

        np.testing.assert_array_almost_equal(loss, brax_loss)

        def strip_arrays(pytree):
            return jnp.concatenate(
                jax.tree_util.tree_map(
                    lambda x: x.flatten(), jax.tree.leaves(pytree),
                ),
            )

        np.testing.assert_array_almost_equal(strip_arrays(grad), strip_arrays(brax_grad))


if __name__ == '__main__':
    absltest.main()
