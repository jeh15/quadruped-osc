from typing import Any
from collections.abc import Callable

import dataclasses

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

# Custom types:
import src.module_types as types
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class TransformerNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


@struct.dataclass
class TransformerConfig:
    input_size: int
    output_size: int
    dtype: Any = jnp.float32
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    qkv_dim: int = 256
    mlp_dim: int = 512
    max_len: int = 512
    dropout_rate: float = 0.3
    attention_dropout_rate: float = 0.3
    kernel_init: Initializer = nn.initializers.xavier_uniform()
    bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
    activation: ActivationFn = nn.elu
    pos_embed_init: Callable | None = None


def sinusoidal_init(
    max_len: int = 2048,
) -> Callable[[jax.Array, tuple[int, ...], Any], jnp.ndarray]:
    def init(key: jax.Array, shape: tuple[int, ...], dtype=np.float32):
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]
        return jnp.array(pe)

    return init


class AddPositionEmbedding(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jnp.ndarray:
        config = self.config
        assert inputs.ndim == 3, (
            f'Number of dimensions should be 3, got {inputs.ndim}'
        )
        length = inputs.shape[-2]
        pos_embed_shape = (1, config.max_len, inputs.shape[-1])
        if config.pos_embed_init is None:
            pos_embedding = sinusoidal_init(max_len=config.max_len)(
                None, pos_embed_shape, None,
            )
        else:
            pos_embedding = self.param(
                'pos_embedding',
                config.pos_embed_init,
                pos_embed_shape,
            )
        pe = pos_embedding[:, :length, :]
        return inputs + pe


class MlpBlock(nn.Module):
    config: TransformerConfig
    out_dim: int

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        config = self.config
        x = nn.Dense(
            features=config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        x = config.activation(x)
        x = nn.Dropout(rate=config.dropout_rate)(
            x, deterministic=deterministic,
        )
        output = nn.Dense(
            features=self.out_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x)
        return output


class EncorderBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs: jax.Array, deterministic: bool) -> jnp.ndarray:
        config = self.config

        # Attention block:
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=deterministic
        )(x)

        x = nn.Dropout(rate=config.dropout_rate)(
            x, deterministic=deterministic,
        )
        x = x + inputs

        # MLP block:
        y = nn.LayerNorm(dtype=config.dtype)(x)
        y = MlpBlock(config=config, out_dim=inputs.shape[-1])(
            y, deterministic=deterministic,
        )
        return x + y


class Transformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(
        self, *, inputs: jax.Array, train: bool,
    ) -> jnp.ndarray:
        config = self.config

        x = nn.DenseGeneral(
            features=(inputs.shape[-1], config.embed_dim),
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        x = nn.Dropout(rate=config.dropout_rate)(
            x, deterministic=not train,
        )
        x = AddPositionEmbedding(config=config)(x)

        for i in range(config.num_layers):
            x = EncorderBlock(config=config)(x, deterministic=not train)

        x = nn.LayerNorm(dtype=config.dtype)(x)
        x = jnp.reshape(x, shape=(-1, inputs.shape[-1] * config.embed_dim))
        logits = nn.Dense(
            features=config.output_size,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x)
        return logits


def make_transformer_network(
    config: TransformerConfig,
    input_normalization_fn: types.InputNormalizationFn =
        types.identity_normalization_fn,
) -> TransformerNetwork:
    """Initializes a transformer network."""
    transformer_network = Transformer(config)

    def apply(normalization_params, params, inputs, train, dropout_key):
        x = input_normalization_fn(inputs, normalization_params)
        return transformer_network.apply(
            params,
            inputs=x,
            train=train,
            rngs={'dropout': dropout_key},
        )

    dummy_input = jnp.zeros((1, config.input_size))
    return TransformerNetwork(
        init=lambda param_key, dropout_key: transformer_network.init(
            {'params': param_key, 'dropout': dropout_key},
            inputs=dummy_input,
            train=True,
        ),
        apply=apply,
    )
