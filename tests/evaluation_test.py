from absl.testing import absltest

import jax
import numpy as np
import flax.linen as nn
import distrax

from brax.envs import fast
from brax.envs.wrappers.training import wrap
import src.algorithms.ppo.network_utilities as ppo_networks
from src.module_types import identity_normalization_fn
from src.distribution_utilities import ParametricDistribution
from brax.training.agents.ppo import networks as brax_ppo_networks

from src.metrics_utilities import Evaluator
from brax.training.acting import Evaluator as BraxEvaluator

jax.config.parse_flags_with_absl()


def rename_params(new_params, params):
    for i, key_name in enumerate(params['params'].keys()):
        new_key_name = f'hidden_{i}'
        new_params['params'][new_key_name] = params['params'][key_name]
    return new_params


class MetricUtilitiesTest(absltest.TestCase):
    def test_evaluation(self):
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
            action_distribution=ParametricDistribution(
                distribution=distrax.Normal,
                bijector=distrax.Tanh(),
            ),
        )
        policy_params = networks.policy_network.init(rng_key)
        value_params = networks.value_network.init(rng_key)
        policy_generator = ppo_networks.make_inference_fn(networks)

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

        brax_policy_generator = brax_ppo_networks.make_inference_fn(brax_networks)

        # Simulation Params:
        num_steps = 10
        num_envs = 2

        evaluator = Evaluator(
            env=env,
            policy_generator=policy_generator,
            num_envs=num_envs,
            episode_length=num_steps,
            action_repeat=1,
            key=rng_key,
        )

        brax_evaluator = BraxEvaluator(
            eval_env=env,
            eval_policy_fn=brax_policy_generator,
            num_eval_envs=num_envs,
            episode_length=num_steps,
            action_repeat=1,
            key=rng_key,
        )

        # Evaluation:
        metrics = evaluator.evaluate((normalization_params, policy_params), {})
        brax_metrics = brax_evaluator.run_evaluation((normalization_params, brax_policy_params), {})

        np.testing.assert_array_almost_equal(
            metrics['eval/episode_reward'],
            brax_metrics['eval/episode_reward'],
        )
        np.testing.assert_array_almost_equal(
            metrics['eval/episode_reward_std'],
            brax_metrics['eval/episode_reward_std'],
        )


if __name__ == '__main__':
    absltest.main()
