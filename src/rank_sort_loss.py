"""Device-agnostic Rank & Sort loss adapted from Oksuz et al., ICCV 2021."""

from __future__ import annotations

import torch


class _RankSort(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        delta: float,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gradients = torch.zeros_like(logits)
        positive_mask = targets > 0
        positive_logits = logits[positive_mask]
        positive_targets = targets[positive_mask]
        positive_count = len(positive_logits)
        if positive_count == 0:
            ctx.save_for_backward(gradients)
            zero = logits.new_zeros(())
            return zero, zero

        threshold = positive_logits.min() - delta
        relevant_negative_mask = (targets == 0) & (logits >= threshold)
        negative_logits = logits[relevant_negative_mask]
        negative_gradient = torch.zeros_like(negative_logits)
        positive_gradient = torch.zeros_like(positive_logits)
        ranking_errors = torch.zeros_like(positive_logits)
        sorting_errors = torch.zeros_like(positive_logits)

        for index in torch.argsort(positive_logits):
            positive_relations = positive_logits - positive_logits[index]
            negative_relations = negative_logits - positive_logits[index]
            if delta > 0:
                positive_relations = torch.clamp(
                    positive_relations / (2.0 * delta) + 0.5,
                    min=0.0,
                    max=1.0,
                )
                negative_relations = torch.clamp(
                    negative_relations / (2.0 * delta) + 0.5,
                    min=0.0,
                    max=1.0,
                )
            else:
                positive_relations = (positive_relations >= 0).to(logits.dtype)
                negative_relations = (negative_relations >= 0).to(logits.dtype)

            positive_rank = positive_relations.sum().clamp_min(eps)
            false_positive_count = negative_relations.sum()
            rank = positive_rank + false_positive_count
            ranking_error = false_positive_count / rank.clamp_min(eps)
            ranking_errors[index] = ranking_error

            current_sorting_error = (
                positive_relations * (1.0 - positive_targets)
            ).sum() / positive_rank
            correct_order = positive_targets >= positive_targets[index]
            target_order = correct_order * positive_relations
            target_positive_rank = target_order.sum().clamp_min(eps)
            target_sorting_error = (
                target_order * (1.0 - positive_targets)
            ).sum() / target_positive_rank
            sorting_error = current_sorting_error - target_sorting_error
            sorting_errors[index] = sorting_error

            if false_positive_count > eps:
                positive_gradient[index] -= ranking_error
                negative_gradient += (
                    negative_relations * ranking_error / false_positive_count
                )

            missorted = (~correct_order) * positive_relations
            missorted_mass = missorted.sum()
            if missorted_mass > eps:
                positive_gradient[index] -= sorting_error
                positive_gradient += missorted * sorting_error / missorted_mass

        gradients[positive_mask] = positive_gradient / positive_count
        gradients[relevant_negative_mask] = negative_gradient / positive_count
        ctx.save_for_backward(gradients)
        return ranking_errors.mean(), sorting_errors.mean()

    @staticmethod
    def backward(ctx, ranking_gradient, sorting_gradient):
        del sorting_gradient
        (gradient,) = ctx.saved_tensors
        return gradient * ranking_gradient, None, None, None


def rank_sort_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank & Sort loss over a flat vector of logits.

    Ranks positives above negatives and sorts them by their continuous target, which
    is what lets localisation quality drive the ranking directly instead of being
    approximated by a classification score.

    Args:
        logits: ``[n]`` raw scores.
        targets: ``[n]`` continuous targets in ``[0, 1]``; zero marks a negative.
        delta: smoothing of the step function used to approximate the rank.

    Returns:
        Tuple of the ranking error and the sorting error; both are zero when the batch
        holds no positive.

    Raises:
        ValueError: if ``logits`` and ``targets`` do not have matching 1-D shapes.
    """
    if logits.ndim != 1 or targets.shape != logits.shape:
        raise ValueError(
            f"RankSort expects matching vectors, got {logits.shape} and {targets.shape}"
        )
    if not torch.any(targets > 0):
        zero = logits.sum() * 0.0
        return zero, zero
    return _RankSort.apply(logits, targets, float(delta), 1e-10)
