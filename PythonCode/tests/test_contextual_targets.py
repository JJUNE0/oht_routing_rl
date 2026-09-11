import math
import unittest

import torch

from oht_routing.algorithms.rl.contextual_td7.targets import (
    CRITIC_TARGET_CDQ,
    CRITIC_TARGET_UBOC,
    UBOC_BETA,
    aggregate_critic_target,
    bellman_target,
    cdq_target_value,
    critic_target_value,
    scale_policy_action,
    target_applied_action,
    uboc_ensemble_statistics,
    uboc_target_value,
    validate_critic_target_mode,
)


class ContextualTargetTests(unittest.TestCase):
    def test_policy_scale_and_raw_policy_space_noise(self):
        policy = torch.tensor([[-1.0], [0.5], [1.0]])
        scaled = scale_policy_action(policy, 0.1)
        torch.testing.assert_close(
            scaled, torch.tensor([[-0.1], [0.05], [0.1]])
        )
        result = target_applied_action(
            policy, action_scale=0.1, noise_std=0.0, noise_clip=0.02,
            noise=torch.tensor([[-1.0], [0.01], [1.0]]),
        )
        torch.testing.assert_close(
            result, torch.tensor([[-0.1], [0.051], [0.1]])
        )

    def test_done_removes_bootstrap(self):
        reward = torch.tensor([[2.0], [3.0]])
        done = torch.tensor([[0.0], [1.0]])
        next_value = torch.tensor([[5.0], [20.0]])
        result = bellman_target(reward, done, next_value, gamma=0.9)
        torch.testing.assert_close(result, torch.tensor([[6.5], [3.0]]))

    def test_nonfinite_target_fails(self):
        with self.assertRaises(FloatingPointError):
            bellman_target(
                torch.tensor([[float("nan")]]), torch.zeros(1, 1),
                torch.zeros(1, 1), gamma=0.99,
            )

    def test_bellman_rejects_per_critic_input(self):
        with self.assertRaisesRegex(ValueError, r"\[B,1\]"):
            bellman_target(
                torch.zeros(3, 1), torch.zeros(3, 1),
                torch.zeros(3, 2), gamma=0.99,
            )


class ClippedDoubleQTests(unittest.TestCase):
    def test_minimum_over_two_critics(self):
        q = torch.tensor([[10.0, 5.0], [20.0, 30.0]])
        torch.testing.assert_close(
            cdq_target_value(q), torch.tensor([[5.0], [20.0]])
        )

    def test_more_than_two_critics_is_refused(self):
        with self.assertRaisesRegex(ValueError, "exactly two critics"):
            cdq_target_value(torch.zeros(4, 3))


class UBOCTargetTests(unittest.TestCase):
    def test_beta_is_one_over_sqrt_pi(self):
        # E[min(X, Y)] = mu - sigma / sqrt(pi) for two i.i.d. normals, which is
        # what makes UBOC share the clipped double-Q expectation. The UD7
        # reference implementation hard-codes the same value.
        self.assertAlmostEqual(UBOC_BETA, 0.5641896, places=7)
        self.assertAlmostEqual(UBOC_BETA, 1.0 / math.sqrt(math.pi), places=12)

    def test_mean_minus_beta_times_unbiased_deviation(self):
        q = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0]])
        mean, deviation = uboc_ensemble_statistics(q)
        torch.testing.assert_close(mean, torch.tensor([[5.0]]))
        # Unbiased (ddof=1) deviation of 1..9 step 2 is sqrt(10).
        torch.testing.assert_close(
            deviation, torch.tensor([[math.sqrt(10.0)]])
        )
        torch.testing.assert_close(
            uboc_target_value(q, beta=0.5),
            torch.tensor([[5.0 - 0.5 * math.sqrt(10.0)]]),
        )

    def test_identical_critics_leave_the_mean_untouched(self):
        q = torch.full((6, 5), -2.5)
        torch.testing.assert_close(
            uboc_target_value(q), torch.full((6, 1), -2.5)
        )

    def test_target_never_exceeds_the_mean(self):
        torch.manual_seed(7)
        q = torch.randn(64, 5) * 3.0
        mean, _ = uboc_ensemble_statistics(q)
        self.assertTrue(bool((uboc_target_value(q) <= mean).all()))

    def test_two_critic_target_sits_above_the_minimum(self):
        # With two critics the deviation is |a - b| / sqrt(2), so the UBOC
        # penalty is beta/sqrt(2) ~= 0.399 of the gap against the 0.5 of the
        # minimum: same expectation, less pessimism per sample.
        q = torch.tensor([[4.0, 10.0]])
        expected = 7.0 - UBOC_BETA * 6.0 / math.sqrt(2.0)
        torch.testing.assert_close(
            uboc_target_value(q), torch.tensor([[expected]])
        )
        self.assertGreater(
            float(uboc_target_value(q)), float(cdq_target_value(q))
        )

    @staticmethod
    def _c4(num_critics):
        """E[unbiased sample deviation] / sigma for normal samples.

        The unbiased estimator is of the *variance*; its square root
        underestimates sigma by this factor, and the shortfall is what keeps
        the finite-N expectation just above the clipped double-Q one.
        """
        return math.sqrt(2.0 / (num_critics - 1)) * (
            math.gamma(num_critics / 2.0)
            / math.gamma((num_critics - 1) / 2.0)
        )

    def test_expectation_approaches_clipped_double_q_as_critics_grow(self):
        # Theorem 4.1 equates E[min of two] with E[mean - beta * deviation],
        # which holds in the limit: at N = 5 the implemented estimator is
        # about 6% less pessimistic than the minimum, closing as N grows.
        mu, sigma = 1.0, 2.0
        cdq_expectation = mu - sigma * UBOC_BETA
        generator = torch.Generator().manual_seed(11)
        errors = []
        for num_critics in (5, 25, 200):
            samples = (
                torch.randn(120_000, num_critics, generator=generator)
                * sigma + mu
            )
            observed = float(uboc_target_value(samples).mean())
            predicted = mu - sigma * UBOC_BETA * self._c4(num_critics)
            self.assertAlmostEqual(observed, predicted, delta=0.02)
            errors.append(abs(observed - cdq_expectation))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[2], 0.02)

    def test_variance_is_lower_than_clipped_double_q_at_every_size(self):
        # Theorem 4.2: the variance gap is strictly positive for all N >= 2,
        # and Corollary 4.3 bounds it by sigma^2 (1 - 1/pi).
        sigma = 2.0
        generator = torch.Generator().manual_seed(13)
        pairs = torch.randn(300_000, 2, generator=generator) * sigma + 1.0
        cdq_variance = float(cdq_target_value(pairs).var())
        bound = sigma ** 2 * (1.0 - 1.0 / math.pi)
        previous = cdq_variance
        for num_critics in (2, 3, 5, 10):
            samples = (
                torch.randn(300_000, num_critics, generator=generator)
                * sigma + 1.0
            )
            variance = float(uboc_target_value(samples).var())
            self.assertLess(variance, cdq_variance)
            self.assertLess(variance, previous)
            self.assertLess(cdq_variance - variance, bound)
            previous = variance

    def test_shape_and_finiteness_contracts(self):
        with self.assertRaisesRegex(ValueError, "N >= 2"):
            uboc_target_value(torch.zeros(3, 1))
        with self.assertRaisesRegex(ValueError, "N >= 2"):
            uboc_target_value(torch.zeros(3))
        with self.assertRaises(FloatingPointError):
            uboc_target_value(torch.full((2, 3), float("inf")))
        with self.assertRaisesRegex(ValueError, "beta"):
            uboc_target_value(torch.zeros(2, 3), beta=-1.0)

    def test_aggregate_reports_the_ensemble_behind_the_value(self):
        q = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0]])
        aggregate = aggregate_critic_target(q, mode=CRITIC_TARGET_UBOC)
        torch.testing.assert_close(
            aggregate.ensemble_mean, torch.tensor([[5.0]])
        )
        torch.testing.assert_close(
            aggregate.ensemble_deviation, torch.tensor([[math.sqrt(10.0)]])
        )
        torch.testing.assert_close(aggregate.minimum, torch.tensor([[1.0]]))
        torch.testing.assert_close(aggregate.value, uboc_target_value(q))
        torch.testing.assert_close(
            aggregate.gap_above_minimum, aggregate.value - aggregate.minimum
        )
        self.assertGreater(float(aggregate.gap_above_minimum), 0.0)

    def test_aggregate_under_cdq_is_the_minimum_with_no_gap(self):
        q = torch.tensor([[4.0, 10.0], [8.0, 2.0]])
        aggregate = aggregate_critic_target(q, mode=CRITIC_TARGET_CDQ)
        torch.testing.assert_close(aggregate.value, aggregate.minimum)
        torch.testing.assert_close(
            aggregate.gap_above_minimum, torch.zeros(2, 1)
        )
        # The statistics are still reported so the two arms log the same keys.
        torch.testing.assert_close(
            aggregate.ensemble_mean, torch.tensor([[7.0], [5.0]])
        )

    def test_aggregate_rejects_cdq_beyond_two_critics(self):
        with self.assertRaisesRegex(ValueError, "exactly two critics"):
            aggregate_critic_target(torch.zeros(2, 5), mode=CRITIC_TARGET_CDQ)

    def test_clipping_per_critic_would_change_the_uboc_aggregate(self):
        # This is why the learner clips the aggregate rather than each critic.
        # A shared clamp commutes with the minimum, so clipped double-Q is
        # indifferent; it compresses the spread the UBOC penalty is built
        # from, so UBOC is not.
        low, high = 2.0, 8.0
        q = torch.tensor([[-4.0, 1.0, 5.0, 9.0, 14.0]])
        clipped_first = uboc_target_value(q.clamp(low, high))
        clipped_after = uboc_target_value(q).clamp(low, high)
        self.assertNotAlmostEqual(
            float(clipped_first), float(clipped_after), places=3
        )

        pair = torch.tensor([[-4.0, 14.0]])
        torch.testing.assert_close(
            cdq_target_value(pair.clamp(low, high)),
            cdq_target_value(pair).clamp(low, high),
        )

    def test_mode_dispatch_and_validation(self):
        q = torch.tensor([[4.0, 10.0]])
        torch.testing.assert_close(
            critic_target_value(q, mode=CRITIC_TARGET_CDQ),
            cdq_target_value(q),
        )
        torch.testing.assert_close(
            critic_target_value(q, mode=CRITIC_TARGET_UBOC),
            uboc_target_value(q),
        )
        self.assertEqual(
            validate_critic_target_mode(CRITIC_TARGET_UBOC),
            CRITIC_TARGET_UBOC,
        )
        with self.assertRaisesRegex(ValueError, "critic_target_mode"):
            critic_target_value(q, mode="minimum")


if __name__ == "__main__":
    unittest.main()
