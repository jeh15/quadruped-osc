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
from src.algorithms.apg.network_utilities import APGNetworkParams

import src.algorithms.apg.network_utilities as apg_networks
import src.algorithms.apg.loss_utilities as loss_utilities
import src.optimization_utilities as optimization_utilities
import src.metrics_utilities as metrics_utilities


InferenceParams = Tuple[running_statistics.NestedMeanStd, types.Params]

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainState:
    opt_state: optax.OptState
    params: APGNetworkParams
    normalization_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def strip_weak_type(pytree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)
    return jax.tree_util.tree_map(f, pytree)


def train(
    environment: envs.Env,
    evaluation_environment: Optional[envs.Env],
    num_epochs: int,
    num_training_steps: int,
    horizon_length: int,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_evaluation_envs: int = 128,
    num_evaluations: int = 1,
    deterministic_evaluation: bool = False,
    reset_per_epoch: bool = False,
    seed: int = 0,
    normalize_observations: bool = True,
    network_factory: types.NetworkFactory[apg_networks.APGNetworks] = apg_networks.make_apg_networks,
    optimizer: optax.GradientTransformation = optax.adam(1e-4),
    loss_function: Callable[..., Tuple[jnp.ndarray, types.Metrics]] =
    loss_utilities.loss_function,
    progress_fn: Callable[[int, int, types.Metrics], None] = lambda *args: None,
    checkpoint_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    wandb: Optional[Any] = None,
    use_float64: bool = False,
):
    training_start_time = time.time()

    # JAX Device management:
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count

    assert num_envs % device_count == 0

    # Training Loop Iteration Parameters:
    num_steps_per_epoch = num_training_steps * num_envs * horizon_length

    # Generate Random Key:
    # key = jax.random.key(seed)
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, env_key, eval_key = jax.random.split(local_key, 3)

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
    step_fn = jax.jit(jax.vmap(env.step))
    envs_key = jax.random.split(env_key, (local_devices_to_use, num_envs // process_count))
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
    make_policy = apg_networks.make_inference_fn(apg_networks=network)

    # Initialize Loss Function:
    # functools.partial loss_function to capture parameters:
    loss_fn = functools.partial(
        loss_function,
        make_policy=make_policy,
        env=env,
        action_repeat=action_repeat,
    )

    gradient_udpate_fn = optimization_utilities.gradient_update_fn(
        loss_fn=loss_fn,
        optimizer=optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
        has_aux=True,
        return_grads=True,
    )

    loss_grad = jax.grad(loss_fn, has_aux=True)
    def clip_by_global_norm(grads):
        g_norm = optax.global_norm(grads)
        max_gradient_norm = 1e9
        trigger = g_norm < max_gradient_norm
        return jax.tree.map(
            lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
            grads,
        )

    def minibatch_step(
        carry,
        unused_t,
    ):
        (opt_state, normalization_params, params, state, key) = carry

        key, subkey = jax.random.split(key)
        # Gradient Update Function computes mean before clipping Brax Clips then computes mean.
        (_, data), params, opt_state, grads = gradient_udpate_fn(
            params,
            normalization_params,
            state,
            subkey,
            opt_state=opt_state,
        )

        # Brax way:
        # grads, data = loss_grad(params, normalization_params, state, subkey)
        # grads = clip_by_global_norm(grads)
        # grads = jax.lax.pmean(grads, axis_name=_PMAP_AXIS_NAME)
        # params_update, opt_state = optimizer.update(grads, opt_state)
        # params = optax.apply_updates(params, params_update)

        # Update Normalization:
        normalization_params = running_statistics.update(
            normalization_params,
            data['observations'],
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        metrics = {
            'grad_norm': optax.global_norm(grads),
            'params_norm': optax.global_norm(params),
        }

        return (opt_state, normalization_params, params, data['state'], key), metrics


    def training_epoch(
        train_state: TrainState,
        state: envs.State,
        key: types.PRNGKey,
    ) -> Tuple[TrainState, envs.State, types.Metrics]:
        (opt_state, normalization_params, params, final_state, key), loss_metrics = jax.lax.scan(
            f=minibatch_step,
            init=(train_state.opt_state, train_state.normalization_params, train_state.params, state, key),
            xs=(),
            length=num_training_steps,
        )
        new_train_state = TrainState(
            opt_state=opt_state,
            params=params,
            normalization_params=normalization_params,
            env_steps=train_state.env_steps + num_steps_per_epoch,
        )
        return new_train_state, final_state, loss_metrics, key

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
        train_state, state, metrics, key = strip_weak_type(result)

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
        return train_state, state, metrics, key

    # Initialize Params and Train State:
    init_params = APGNetworkParams(
        policy_params=network.policy_network.init(global_key),
    )
    del global_key

    # Can't pass optimizer function to device_put_replicated:
    dtype = 'float64' if use_float64 else 'float32'
    train_state = TrainState(
        opt_state=optimizer.init(init_params),
        params=init_params,
        normalization_params=running_statistics.init_state(
            specs.Array(env_state.obs.shape[-1:], jnp.dtype(dtype))
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
            train_state.params.policy_params,
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

    local_key, epoch_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    
    # Training Loop:
    for epoch_iteration in range(num_epochs):
        # Logging:
        logging.info(
            'starting iteration %s %s', epoch_iteration, time.time() - training_start_time,
        )

        # Epoch Training Iteration:
        (train_state, env_state, training_metrics, epoch_keys) = (
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
                train_state.params.policy_params,
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
        train_state.params.policy_params,
    ))
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
