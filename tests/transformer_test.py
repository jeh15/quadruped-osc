from absl.testing import absltest

import jax

from src.transformer_network import make_transformer_network, TransformerConfig

jax.config.parse_flags_with_absl()


class TransformerTest(absltest.TestCase):
    def test_transfomer_network(self):
        config = TransformerConfig(input_size=512, output_size=10)
        param_key = jax.random.PRNGKey(0)
        dropout_key = jax.random.PRNGKey(1)
        rng = jax.random.PRNGKey(2)

        network = make_transformer_network(config)

        params = network.init(param_key, dropout_key)

        x = jax.random.normal(rng, (1, config.input_size))
        logits = network.apply(
            params, inputs=x, train=True, dropout_key=dropout_key,
        )

        print(logits.shape)
        print(logits)


if __name__ == '__main__':
    absltest.main()
