from typing import Any, Tuple, Callable

import jax
import jax.numpy as jnp
from brax import envs

from src.algorithms.gb_ppo import network_utilities as gb_ppo_networks
from src import module_types as types


def calculate_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    truncation_mask: jnp.ndarray,
    termination_mask: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculates the Generalized Advantage Estimation."""
    values_ = jnp.concatenate(
        [values, jnp.expand_dims(bootstrap_value, axis=0)], axis=0,
    )
    deltas = rewards + gamma * termination_mask * values_[1:] - values_[:-1]
    deltas *= truncation_mask

    initial_gae = jnp.zeros_like(bootstrap_value)

    def scan_loop(carry, xs):
        gae = carry
        truncation_mask, termination_mask, delta = xs
        gae = (
            delta
            + gamma * gae_lambda * termination_mask * truncation_mask * gae
        )
        return gae, gae

    _, vs = jax.lax.scan(
        scan_loop,
        initial_gae,
        (truncation_mask, termination_mask, deltas),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    vs = jnp.add(vs, values)
    vs_ = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, axis=0)], axis=0,
    )
    advantages = (
        rewards
        + gamma * termination_mask * vs_ - values
    ) * truncation_mask
    return vs, advantages


def policy_loss_function(
    policy_params: types.Params,
    value_params: types.Params,
    normalization_params: Any,
    state: envs.State,
    rng_key: types.PRNGKey,
    make_policy: Callable[..., Any],
    gb_ppo_networks: gb_ppo_networks.GBPPONetworks,
    rollout_fn: Callable[..., Any],
    rollout_length: int,
    clip_coef = 0.3,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
) -> Tuple[jnp.ndarray, Tuple[envs.State, types.Transition, types.Metrics]]:
    # Unpack GBPPO networks:
    action_distribution = gb_ppo_networks.action_distribution
    policy_apply = gb_ppo_networks.policy_network.apply
    value_apply = gb_ppo_networks.value_network.apply

    # Split Key:
    new_rng_key, entropy_key, rollout_key = jax.random.split(rng_key, 3)

    # Generate Episode Data: (Use same RNG Key for same rollout)
    policy_fn = make_policy((normalization_params, policy_params))
    def f(carry, unused_t):
        current_state, key = carry
        key, subkey = jax.random.split(key)
        next_state, data = rollout_fn(
            state=current_state,
            policy=policy_fn,
            key=key,
        )
        return (next_state, subkey), data

    (state, _), data = jax.lax.scan(
        f,
        (state, rollout_key),
        (),
        length=rollout_length,
    )

    # Combine into one leading dimension (batch_size * num_minibatches, unroll_length) (bld = batch leading dimension)
    data = jax.tree.map(lambda x: jnp.swapaxes(x, 1, 2), data)
    bld_data = jax.tree.map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)

    # Reorder data: (B, T, ...) -> (T, B, ...) (tld = time leading dimension)
    tld_data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), bld_data)

    logits = policy_apply(
        normalization_params, policy_params, tld_data.observation,
    )
    values = value_apply(
        normalization_params, value_params, tld_data.observation,
    )
    bootstrap_values = value_apply(
        normalization_params, value_params, tld_data.next_observation[-1],
    )

    # Create masks for truncation and termination: (We can also mask the moment of contact)
    rewards = tld_data.reward
    truncation_mask = 1 - tld_data.extras['state_data']['truncation']

    # This formulation does not make sense...
    # termination_mask = (1 - tld_data.termination) * truncation_mask
    termination_mask = 1 - tld_data.termination

    # Calculate GAE:
    _, advantages = calculate_gae(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_values,
        truncation_mask=truncation_mask,
        termination_mask=termination_mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    if normalize_advantages:
        advtanages_mean = jax.lax.stop_gradient(jnp.mean(advantages))
        advantages_std = jax.lax.stop_gradient(jnp.std(advantages))
        advantages = (
            (advantages - advtanages_mean) / (advantages_std + 1e-8)
        )

    # Sum of advantages:
    # policy_loss = -jnp.sum(advantages)
    policy_loss = -jnp.mean(advantages)

    # Entropy Loss:
    entropy = action_distribution.entropy(
        logits,
        entropy_key,
    )
    entropy_loss = -entropy_coef * jnp.mean(
        entropy,
    )

    loss = policy_loss + entropy_loss

    return loss, (
        state,
        bld_data,
        new_rng_key,
        {
            "total_loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
        },
    )

def value_loss_function(
    value_params: types.Params,
    normalization_params: Any,
    data: types.Transition,
    gb_ppo_networks: gb_ppo_networks.GBPPONetworks,
    value_coef: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, types.Metrics]:
    # Unpack GBPPO networks:
    value_apply = gb_ppo_networks.value_network.apply

    # Reorder data: (B, T, ...) -> (T, B, ...)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    values = value_apply(
        normalization_params, value_params, data.observation,
    )
    bootstrap_values = value_apply(
        normalization_params, value_params, data.next_observation[-1],
    )

    # Create masks for truncation and termination:
    rewards = data.reward
    truncation_mask = 1 - data.extras['state_data']['truncation']

    # This formulation does not make sense...
    # termination_mask = (1 - data.termination) * truncation_mask
    termination_mask = 1 - data.termination

    # Calculate GAE:
    vs, _ = calculate_gae(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_values,
        truncation_mask=truncation_mask,
        termination_mask=termination_mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # Value Loss:
    target_values = jax.lax.stop_gradient(vs)
    # target_values = vs
    value_loss = value_coef * jnp.mean(
        jnp.square(target_values - values),
    )

    loss = value_loss

    return loss, {
        "value_loss": value_loss,
    }