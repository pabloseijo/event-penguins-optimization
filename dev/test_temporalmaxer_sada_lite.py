"""Unit tests for class-balanced semantic adaptation helpers."""

from __future__ import annotations

import unittest

import torch

from dev.train_temporalmaxer_sada_lite_cv import (
    DomainDiscriminator,
    GradientReverse,
    domain_loss,
    target_pseudo_masks,
)


class TemporalMaxerSadaLiteTest(unittest.TestCase):
    def test_pseudo_masks_follow_source_prior_and_are_disjoint(self) -> None:
        scores = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        output = {"classification_logits": [scores]}
        valid = torch.ones(2, 10, dtype=torch.bool)
        action, background = target_pseudo_masks(output, valid, 0.1, 0.5)
        self.assertEqual(int(action.sum()), 2)
        self.assertEqual(int(background.sum()), 10)
        self.assertFalse((action & background).any())
        self.assertTrue(action[1, -2:].all())

    def test_gradient_reverse_changes_only_backward_sign_and_scale(self) -> None:
        value = torch.tensor([1.0, 2.0], requires_grad=True)
        reversed_value = GradientReverse.apply(value, 0.5)
        self.assertTrue(torch.equal(value, reversed_value))
        reversed_value.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.tensor([-0.5, -0.5])))

    def test_domain_loss_accepts_a_batch_without_source_actions(self) -> None:
        discriminator = DomainDiscriminator(4)
        source = torch.empty(0, 4, requires_grad=True)
        target = torch.randn(3, 4, requires_grad=True)
        loss = domain_loss(discriminator, source, target, beta=0.5, target_weight=0.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(target.grad).all())


if __name__ == "__main__":
    unittest.main()
