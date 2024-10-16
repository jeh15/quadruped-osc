import functools
import time
from typing import Any, Callable, Optional, Tuple

from absl import logging
import flax.struct
import jax
import jax.numpy as jnp
import flax
import optax
import orbax.checkpoint as ocp
import numpy as np

from brax import base
from brax import envs
from brax.envs.wrappers.training import wrap
from brax.training.acme import running_statistics, specs
from brax.training import pmap
import src.module_types as types
from src.algorithms.gb_ppo.network_utilities import GBPPONetworkParams

import src.algorithms.gb_ppo.network_utilities as gb_ppo_networks
import src.algorithms.gb_ppo.loss_utilities as loss_utilities
import src.optimization_utilities as optimization_utilities
import src.training_utilities as trainining_utilities
import src.metrics_utilities as metrics_utilities


InferenceParams = Tuple[running_statistics.NestedMeanStd, types.Params]

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainState:
    policy_opt_state: optax.OptState
    value_opt_state: optax.OptState
    policy_params: types.Params
    value_params: types.Params
    target_value_params: types.Params
    normalization_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def unpmap(v):
    return jax.tree.map(lambda x: x[0], v)


def strip_weak_type(pytree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)
    return jax.tree.map(f, pytree)


def train(
    environment: envs.Env,
    evaluation_environment: Optional[envs.Env],
    num_epochs: int,
    num_training_steps: int,
    episode_length: int,
    horizon_length: int = 32,
    tau: float = 0.005,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_evaluation_envs: int = 128,
    num_evaluations: int = 1,
    deterministic_evaluation: bool = False,
    reset_per_epoch: bool = False,
    seed: int = 0,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_ppo_iterations: int = 4,
    normalize_observations: bool = True,
    network_factory: types.NetworkFactory[gb_ppo_networks.GBPPONetworks] = gb_ppo_networks.make_gb_ppo_networks,
    policy_optimizer: optax.GradientTransformation = optax.adam(1e-4),
    value_optimizer: optax.GradientTransformation = optax.adam(1e-4),
    policy_loss_function: Callable[..., Tuple[jnp.ndarray, types.Metrics]] =
    loss_utilities.policy_loss_function,
    value_loss_function: Callable[..., Tuple[jnp.ndarray, types.Metrics]] =
    loss_utilities.value_loss_function,
    progress_fn: Callable[[int, int, types.Metrics], None] = lambda *args: None,
    checkpoint_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    wandb: Optional[Any] = None,
):
    assert batch_size * num_minibatches % num_envs == 0
    training_start_time = time.time()

    # JAX Device management:
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count

    assert num_envs % device_count == 0

    # Training Loop Iteration Parameters:
    num_steps_per_train_step = (
        batch_size * num_minibatches * horizon_length * action_repeat
    )
    num_steps_per_epoch = (
        num_steps_per_train_step * num_training_steps
    )

    # Generate Random Key:
    # key = jax.random.key(seed)
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, env_key, eval_key = jax.random.split(local_key, 3)
    policy_key, value_key = jax.random.split(global_key)
    del global_key

    # Initialize Environment:
    _randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
        randomization_key = jax.random.split(env_key, randomization_batch_size)
        _randomization_fn = functools.partial(
            randomization_fn, rng=randomization_key,
        )

    env = wrap(
        env=environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=_randomization_fn,
    )

    # vmap for multiple devices:
    reset_fn = jax.jit(jax.vmap(env.reset))
    envs_key = jax.random.split(env_key, num_envs // process_count)
    envs_key = jnp.reshape(
        envs_key, (local_devices_to_use, -1) + envs_key.shape[1:],
    )
    env_state = reset_fn(envs_key)

    # Initialize Normalization Function:
    normalization_fn = lambda x, y: x
    if normalize_observations:
        normalization_fn = running_statistics.normalize

    # Initialize Network:
    # functools.partial network_factory to capture parameters:
    network = network_factory(
        observation_size=env_state.obs.shape[-1],
        action_size=env.action_size,
        input_normalization_fn=normalization_fn,
    )
    make_policy = gb_ppo_networks.make_inference_fn(gb_ppo_networks=network)

    # Initialize Loss Function:
    # functools.partial loss_function to capture parameters:
    rollout_fn = functools.partial(
        trainining_utilities.unroll_policy_steps,
        env=env,
        num_steps=horizon_length,
        extra_fields=('truncation',),
    )
    policy_loss_fn = functools.partial(
        policy_loss_function,
        make_policy=make_policy,
        gb_ppo_networks=network,
        rollout_fn=rollout_fn,
        rollout_length=batch_size * num_minibatches // num_envs,
    )

    value_loss_fn = functools.partial(
        value_loss_function,
        gb_ppo_networks=network,
    )

    policy_gradient_update_fn = optimization_utilities.gradient_update_fn(
        loss_fn=policy_loss_fn,
        optimizer=policy_optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
        has_aux=True,
        return_grads=True,
    )
    policy_gradient_update_fn = jax.jit(policy_gradient_update_fn)

    value_gradient_update_fn = optimization_utilities.gradient_update_fn(
        loss_fn=value_loss_fn,
        optimizer=value_optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
        has_aux=True,
        return_grads=True,
    )
    value_gradient_update_fn = jax.jit(value_gradient_update_fn)


    def minibatch_step(
        carry,
        data: types.Transition,
        normalization_params: running_statistics.RunningStatisticsState,
    ):
        opt_state, params = carry
        (_, metrics), params, opt_state, grads = value_gradient_update_fn(
            params,
            normalization_params,
            data,
            opt_state=opt_state,
        )

        return (opt_state, params), (metrics, grads)

    def sgd_step(
        carry,
        unusted_t,
        data: types.Transition,
        normalization_params: running_statistics.RunningStatisticsState,
    ):
        opt_state, params, key = carry
        key, permutation_key = jax.random.split(key, 2)

        # Shuffle Data:
        def permute_data(x: jnp.ndarray):
            x = jax.random.permutation(permutation_key, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree.map(permute_data, data)
        (opt_state, params), (metrics, grads) = jax.lax.scan(
            functools.partial(
                minibatch_step, normalization_params=normalization_params,
            ),
            (opt_state, params),
            shuffled_data,
            length=num_minibatches,
        )

        return (opt_state, params, key), (metrics, grads)

    def training_step(
        carry: Tuple[TrainState, envs.State, types.PRNGKey],
        unused_t,
    ) -> Tuple[Tuple[TrainState, envs.State, types.PRNGKey], types.Metrics]:
        train_state, initial_state, key = carry

        next_key, rollout_key, sgd_key = jax.random.split(key, 3)

        (_, (state, data, _, policy_metrics)), policy_params, policy_opt_state, policy_grads = policy_gradient_update_fn(
                train_state.policy_params,
                train_state.value_params,
                train_state.normalization_params,
                initial_state,
                key,
                opt_state=train_state.policy_opt_state,
            )

        # Use Extra Data for Critic:
        policy_fn = make_policy((
            train_state.normalization_params, policy_params,
        ))

        # Generate additional Episode Data for Critic:
        def f(carry, unused_t):
            current_state, key = carry
            key, subkey = jax.random.split(key)
            next_state, data = trainining_utilities.unroll_policy_steps(
                env=env,
                state=current_state,
                policy=policy_fn,
                key=key,
                num_steps=horizon_length,
                extra_fields=('truncation',),
            )
            return (next_state, subkey), data

        (_, _), unroll_data = jax.lax.scan(
            f,
            (initial_state, rollout_key),
            (),
            length=256 * 32 // num_envs,
        )

        # Swap leading dimensions: (T, B, ...) -> (B, T, ...)
        unroll_data = jax.tree.map(lambda x: jnp.swapaxes(x, 1, 2), unroll_data)
        data = jax.tree.map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), unroll_data,
        )

        # Update Normalization:
        normalization_params = running_statistics.update(
            train_state.normalization_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        (value_opt_state, value_params, _), (metrics, value_grads) = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalization_params=normalization_params,
            ),
            (train_state.value_opt_state, train_state.value_params, sgd_key),
            (),
            length=num_ppo_iterations,
        )

        target_value_params = jax.tree.map(
            lambda x, y: (1 - tau) * x + tau * y,
            train_state.target_value_params,
            value_params,
        )

        # Train State:
        new_train_state = TrainState(
            policy_opt_state=policy_opt_state,
            value_opt_state=value_opt_state,
            policy_params=policy_params,
            value_params=value_params,
            target_value_params=target_value_params,
            normalization_params=normalization_params,
            env_steps=train_state.env_steps + num_steps_per_train_step,
        )

        grads = {
            'policy_grads': optax.global_norm(policy_grads),
            'value_grads': optax.global_norm(value_grads),
        }
        metrics.update(policy_metrics)
        metrics.update(grads)

        return (new_train_state, state, next_key), metrics

    def training_epoch(
        train_state: TrainState,
        state: envs.State,
        key: types.PRNGKey,
    ) -> Tuple[TrainState, envs.State, types.Metrics]:
        (train_state, state, _), metrics = jax.lax.scan(
            training_step,
            (train_state, state, key),
            (),
            length=num_training_steps,
        )
        return train_state, state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    def training_epoch_with_metrics(
        train_state: TrainState,
        state: envs.State,
        key: types.PRNGKey,
    ) -> Tuple[TrainState, envs.State, types.Metrics]:
        # I would like to get rid of this:
        nonlocal training_walltime
        start_time = time.time()
        train_state, state = strip_weak_type((train_state, state))
        result = training_epoch(train_state, state, key)
        train_state, state, metrics = strip_weak_type(result)

        metrics = jax.tree.map(jnp.mean, metrics)
        jax.tree.map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - start_time
        training_walltime += epoch_training_time
        steps_per_second = num_steps_per_epoch / epoch_training_time
        metrics = {
            'training/steps_per_second': steps_per_second,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        return train_state, state, metrics

    # Initialize Params and Train State:
    initial_policy_params = network.policy_network.init(policy_key)
    initial_value_params = network.value_network.init(value_key)

    train_state = TrainState(
        policy_opt_state=policy_optimizer.init(initial_policy_params),
        value_opt_state=value_optimizer.init(initial_value_params),
        policy_params=initial_policy_params,
        value_params=initial_value_params,
        target_value_params=initial_value_params,
        normalization_params=running_statistics.init_state(
            specs.Array(env_state.obs.shape[-1:], jnp.dtype('float32'))
        ),
        env_steps=0,
    )

    train_state = jax.device_put_replicated(
        train_state,
        jax.local_devices()[:local_devices_to_use],
    )

    # Setup Evaluation Environment:
    eval_randomization_fn = None
    if randomization_fn is not None:
        eval_randomization_key = jax.random.split(
            eval_key, num_evaluation_envs,
        )
        eval_randomization_fn = functools.partial(
            randomization_fn,
            rng=eval_randomization_key,
        )

    # Must be a separate object or JAX tries to resuse JIT from training environment:
    eval_env = wrap(
        env=evaluation_environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=eval_randomization_fn,
    )

    evaluator = metrics_utilities.Evaluator(
        env=eval_env,
        policy_generator=functools.partial(
            make_policy, deterministic=deterministic_evaluation,
        ),
        num_envs=num_evaluation_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Initialize Metrics:
    metrics = {}
    if process_id == 0 and num_evaluations != 0:
        params = unpmap((
            train_state.normalization_params,
            train_state.policy_params,
        ))
        metrics = evaluator.evaluate(
            policy_params=params,
            training_metrics={},
        )
        logging.info(metrics)
        if wandb is not None:
            wandb.log(metrics)
        progress_fn(0, 0, metrics)
        if checkpoint_fn is not None:
            _train_state = unpmap(train_state)
            checkpoint_fn(iteration=0, train_state=_train_state)

    training_metrics = {}
    training_walltime = 0
    current_step = 0

    # Training Loop:
    for epoch_iteration in range(num_epochs):
        # Logging:
        logging.info(
            'starting iteration %s %s', epoch_iteration, time.time() - training_start_time,
        )

        # Epoch Training Iteration:
        local_key, epoch_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (train_state, env_state, training_metrics) = (
            training_epoch_with_metrics(train_state, env_state, epoch_keys)
        )

        current_step = int(unpmap(train_state.env_steps))

        # If reset per epoch else Auto Reset:
        if reset_per_epoch:
            envs_key = jax.vmap(
                lambda x, s: jax.random.split(x[0], s),
                in_axes=(0, None),
            )(envs_key, envs_key.shape[1])
            env_state = reset_fn(envs_key)

        if process_id == 0:
            # Run Evaluation:
            params = unpmap((
                train_state.normalization_params,
                train_state.policy_params,
            ))
            metrics = evaluator.evaluate(
                policy_params=params,
                training_metrics=training_metrics,
            )
            logging.info(metrics)
            if wandb is not None:
                wandb.log(metrics)
            progress_fn(epoch_iteration+1, current_step, metrics)
            # Save Checkpoint:
            if checkpoint_fn is not None:
                _train_state = unpmap(train_state)
                checkpoint_fn(
                    iteration=epoch_iteration+1, train_state=_train_state,
                )

    total_steps = current_step

    # pmap:
    pmap.assert_is_replicated(train_state)
    params = unpmap((
        train_state.normalization_params,
        train_state.policy_params,
    ))
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)

