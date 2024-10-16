from typing import Tuple, Sequence

import jax
import jax.numpy as jnp
import distrax

import flax
import flax.struct
from flax import linen as nn

from src import networks
from src import transformer_network
import src.distribution_utilities as distribution_utilities
from src import module_types as types
from src.module_types import PRNGKey


@flax.struct.dataclass
class PPONetworks:
    policy_network: transformer_network.TransformerNetwork
    value_network: (
        networks.FeedForwardNetwork | transformer_network.TransformerNetwork
    )
    action_distribution: distribution_utilities.ParametricDistribution


@flax.struct.dataclass
class PPONetworkParams:
    policy_params: types.Params
    value_params: types.Params


def make_inference_fn(ppo_networks: PPONetworks):
    """Creates the params and inference function for the PPO networks."""

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        action_distribution = ppo_networks.action_distribution

        def policy(
            x: types.Observation,
            key: PRNGKey,
        ) -> Tuple[types.Action, types.PolicyData]:
            key, dropout_key, distribution_key = jax.random.split(key, 3)
            logits = policy_network.apply(
                *params,
                inputs=x,
                train=not deterministic,
                dropout_key=dropout_key,
            )
            if deterministic:
                actions = action_distribution.mode(logits)
                policy_data = {}
            else:
                raw_actions = action_distribution.base_distribution_sample(
                    logits, distribution_key
                )
                log_prob = action_distribution.log_prob(logits, raw_actions)
                actions = action_distribution.process_sample(raw_actions)
                policy_data = {"log_prob": log_prob, "raw_action": raw_actions}
            return actions, policy_data

        return policy

    return make_policy


def make_ppo_networks(
    observation_size: int,
    action_size: int,
    input_normalization_fn: types.InputNormalizationFn = types
    .identity_normalization_fn,
    value_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = nn.swish,
    kernel_init: types.Initializer = jax.nn.initializers.lecun_uniform(),
    action_distribution: distribution_utilities.ParametricDistribution = distribution_utilities
    .ParametricDistribution(distribution=distrax.Normal, bijector=distrax.Tanh())
) -> PPONetworks:
    """Create Policy and Value Networks for PPO."""
    # TODO(jeh15): Move Config out to be a partial function:
    config = transformer_network.TransformerConfig(
        input_size=observation_size,
        output_size=2*action_size,
        dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32,
    )
    policy_network = transformer_network.make_transformer_network(
        config, input_normalization_fn,
    )

    value_network = networks.make_value_network(
        input_size=observation_size,
        input_normalization_fn=input_normalization_fn,
        layer_sizes=value_layer_sizes,
        activation=activation,
        kernel_init=kernel_init,
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=action_distribution,
    )
