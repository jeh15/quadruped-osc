from typing import Any, Tuple, Callable
import functools

import jax
import jax.numpy as jnp
from brax import envs

from src.algorithms.apg import network_utilities as apg_networks
from src import module_types as types


def env_step(
    carry: Tuple[envs.State, types.PRNGKey],
    xs: None,
    policy: types.Policy,
    env: envs.Env,
):
    state, key = carry
    key, subkey = jax.random.split(key)
    actions, _ = policy(state.obs, subkey)
    next_state = env.step(state, actions)
    return (next_state, subkey), (next_state.reward, next_state.obs)


def loss_function(
    params: apg_networks.APGNetworkParams,
    normalization_params: Any,
    env_state: envs.State,
    rng_key: types.PRNGKey,
    make_policy: Callable[..., types.PolicyParams],
    env: envs.Env,
    horizon_length: int,
    action_repeat: int,
) -> Tuple[jnp.ndarray, types.Metrics]:
    f = functools.partial(
        env_step, policy=make_policy((normalization_params, params.policy_params)), env=env,
    )
    (final_state, _), (rewards, obs) = jax.lax.scan(
        f=f,
        init=(env_state, rng_key),
        xs=(),
        length=horizon_length / action_repeat,
    )
    loss = -jnp.mean(rewards)

    return loss, {
            "loss": loss,
            "rewards": rewards,
            "observations": obs,
            "state": final_state,
        }
