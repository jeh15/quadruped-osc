from absl.testing import absltest

import jax
import numpy as np
import distrax

from brax.training.distribution import NormalTanhDistribution
from src.distribution_utilities import ParametricDistribution

jax.config.parse_flags_with_absl()


class DistributionTest(absltest.TestCase):
    def test_distribution_fn(self):
        rng_key = jax.random.key(seed=42)
        shape = (12,)

        # Params:
        test_input = jax.random.normal(rng_key, shape)
        std = 0.001
        var = 1.0

        # Brax Distribution:
        brax_distribution = NormalTanhDistribution(
            event_size=shape,
            min_std=std,
        )

        # Distrax Distribution:
        distrax_distribution = ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tanh(),
            min_std=std,
            var_scale=var,
        )

        # Calculate Test Values:
        brax_raw_actions = brax_distribution.sample_no_postprocessing(test_input, rng_key)
        brax_log_prob = brax_distribution.log_prob(test_input, brax_raw_actions)
        brax_postprocessed_actions = brax_distribution.postprocess(brax_raw_actions)

        distrax_raw_actions = distrax_distribution.base_distribution_sample(test_input, rng_key)
        distrax_log_prob = distrax_distribution.log_prob(test_input, distrax_raw_actions)
        distrax_postprocessed_actions = distrax_distribution.process_sample(distrax_raw_actions)

        # Tests:
        np.testing.assert_array_almost_equal(brax_raw_actions, distrax_raw_actions)
        np.testing.assert_array_almost_equal(brax_log_prob, distrax_log_prob)
        np.testing.assert_array_almost_equal(brax_postprocessed_actions, distrax_postprocessed_actions)


if __name__ == '__main__':
    absltest.main()
