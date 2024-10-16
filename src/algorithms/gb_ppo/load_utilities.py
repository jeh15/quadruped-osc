import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import flax.linen as nn
import optax
import distrax
from brax.training.acme import running_statistics, specs
from brax.envs.base import Env

from src.algorithms.gb_ppo import checkpoint_utilities
from src.algorithms.gb_ppo import network_utilities as gb_ppo_networks
from src.algorithms.gb_ppo.train import TrainState
from src.distribution_utilities import ParametricDistribution


def load_policy(checkpoint_name: str, environment: Env):
    # Load Metadata:
    checkpoint_direrctory = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        f"checkpoints/{checkpoint_name}",
    )
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=checkpoint_utilities.default_checkpoint_options(),
        item_names=(
            'network_metadata',
            'loss_metadata',
            'training_metadata',
        ),
    )

    metadata = checkpoint_utilities.load_checkpoint(
        manager=manager,
        network_metadata=checkpoint_utilities.empty_network_metadata(),
        loss_metadata=checkpoint_utilities.empty_loss_metadata(),
        training_metadata=checkpoint_utilities.empty_training_metadata(),
    )
    network_metadata = metadata.network_metadata
    loss_metadata = metadata.loss_metadata
    training_metadata = metadata.training_metadata

    env = environment

    # Restore Structures:
    policy_layer_sizes = (network_metadata.policy_layer_size,) * network_metadata.policy_depth
    value_layer_sizes = (network_metadata.value_layer_size,) * network_metadata.value_depth
    if training_metadata.normalize_observations:
        normalization_fn = running_statistics.normalize
    else:
        normalization_fn = lambda x, y: x

    network = gb_ppo_networks.make_gb_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        input_normalization_fn=normalization_fn,
        policy_layer_sizes=policy_layer_sizes,
        value_layer_sizes=value_layer_sizes,
        activation=eval(network_metadata.activation),
        kernel_init=eval(network_metadata.kernel_init),
        action_distribution=eval(network_metadata.action_distribution),
    )

    policy_optimizer= eval(training_metadata.policy_optimizer)
    value_optimizer = eval(training_metadata.value_optimizer)

    key = jax.random.key(training_metadata.seed)
    initial_policy_params = network.policy_network.init(key)
    initial_value_params = network.value_network.init(key)
    
    train_state = TrainState(
        policy_opt_state=policy_optimizer.init(initial_policy_params),
        value_opt_state=value_optimizer.init(initial_value_params),
        policy_params=initial_policy_params,
        value_params=initial_value_params,
        target_value_params=initial_value_params,
        normalization_params=running_statistics.init_state(
            specs.Array(env.observation_size, jnp.dtype('float32'))
        ),
        env_steps=0,
    )

    # Restore Train State:
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=checkpoint_utilities.default_checkpoint_options(),
        item_names=(
            'train_state',
        ),
    )
    restored_train_state = checkpoint_utilities.load_checkpoint(
        manager=manager,
        train_state=train_state,
    )
    train_state = restored_train_state.train_state

    # Construct Policy:
    make_policy = gb_ppo_networks.make_inference_fn(gb_ppo_networks=network)
    params = (
        train_state.normalization_params, train_state.policy_params,
    )

    return make_policy, params
