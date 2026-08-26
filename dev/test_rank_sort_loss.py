"""Tests for the Rank & Sort loss.

Checks that a correct ordering scores below a reversed one and that a batch
without positives returns a differentiable zero rather than a NaN.
"""

from __future__ import annotations

import unittest

import torch

from src.rank_sort_loss import rank_sort_loss


class RankSortLossTest(unittest.TestCase):
    def test_perfect_order_has_lower_error_than_reversed_order(self) -> None:
        targets = torch.tensor([0.9, 0.6, 0.3, 0.0, 0.0])
        perfect = torch.tensor([3.0, 2.0, 1.0, -1.0, -2.0], requires_grad=True)
        reversed_scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        perfect_error = sum(rank_sort_loss(perfect, targets))
        reversed_error = sum(rank_sort_loss(reversed_scores, targets))
        self.assertLess(float(perfect_error), float(reversed_error))
        reversed_error.backward()
        self.assertTrue(torch.isfinite(reversed_scores.grad).all())
        self.assertTrue(torch.any(reversed_scores.grad != 0))

    def test_empty_positive_batch_is_differentiable_zero(self) -> None:
        logits = torch.randn(5, requires_grad=True)
        error = sum(rank_sort_loss(logits, torch.zeros(5)))
        torch.testing.assert_close(error, torch.zeros_like(error))
        error.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


if __name__ == "__main__":
    unittest.main()
