"""Temporal IoU and non-maximum suppression for 1-D action detections.

Detections are ``[n, 3]`` arrays of ``[t_start, t_end, score]``. Times may be in
seconds or microseconds as long as one run stays consistent.
"""

import numpy as np


def temporal_iou(proposal_min, proposal_max, gt_min, gt_max):
    """Intersection over union between one segment and an array of segments.

    Args:
        proposal_min: start of the candidate segments.
        proposal_max: end of the candidate segments.
        gt_min: start of the reference segment.
        gt_max: end of the reference segment.

    Returns:
        tIoU in ``[0, 1]``, broadcast over the candidate array.
    """
    len_anchors = proposal_max - proposal_min
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.0)
    union_len = len_anchors - inter_len + gt_max - gt_min
    return np.divide(inter_len, union_len)


def temporal_nms(detections: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression over temporal detections.

    Args:
        detections: ``[n, 3]`` array of ``[t_start, t_end, score]``.
        threshold: candidates overlapping a kept detection above this tIoU are dropped.

    Returns:
        The surviving rows of ``detections``, ordered by decreasing score.
    """
    starts = detections[:, 0]
    ends = detections[:, 1]
    scores = detections[:, 2]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = temporal_iou(starts[order[1:]], ends[order[1:]], starts[i], ends[i])
        order = order[np.where(ious <= threshold)[0] + 1]
    return detections[keep, :]


def temporal_soft_nms(
    detections: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> np.ndarray:
    """Soft-NMS with gaussian decay for 1-D temporal detections.

    Overlapping detections have their score decayed instead of being removed, so a
    display adjacent in time to a higher-scoring one survives instead of being
    erased by hard suppression.

    Args:
        detections: ``[n, 3]`` array of ``[t_start, t_end, score]``.
        sigma: width of the gaussian decay; smaller values suppress harder.
        score_threshold: detections decayed below this score are dropped.

    Returns:
        Surviving detections with updated scores, ordered by decreasing score.
    """
    dets = detections.copy()
    scores = dets[:, 2].copy()

    indices = list(range(len(dets)))
    keep = []

    while indices:
        best_local = int(np.argmax(scores[indices]))
        best = indices[best_local]
        keep.append(best)
        indices.pop(best_local)

        for idx in indices:
            iou = temporal_iou(
                dets[idx, 0], dets[idx, 1],
                dets[best, 0], dets[best, 1],
            )
            scores[idx] *= np.exp(-(iou ** 2) / sigma)

    result = dets[keep]
    result[:, 2] = scores[keep]
    result = result[result[:, 2] >= score_threshold]
    return result[result[:, 2].argsort()[::-1]]
