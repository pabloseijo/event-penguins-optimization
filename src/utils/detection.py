import numpy as np


def temporal_iou(proposal_min, proposal_max, gt_min, gt_max):
    """
    Compute IoU score between a groundtruth bbox and the proposals.

    Args:
        proposal_min: List of temporal anchor min.
        proposal_max: List of temporal anchor max.
        gt_min: Groundtruth temporal box min.
        gt_max: Groundtruth temporal box max.

    Returns:
        list[float]: List of iou scores.
    """
    len_anchors = proposal_max - proposal_min
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.0)
    union_len = len_anchors - inter_len + gt_max - gt_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def temporal_nms(detections: np.ndarray, threshold: float) -> np.ndarray:
    """
    Perform 1D non-maximum suppression on n detections

    Args:
        detections: Detection results before NMS (n x 3).
                    Each detection has form (t_start, t_end, score)
        threshold: Threshold of NMS.

    Returns:
        Detection results after NMS.
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
        idxs = np.where(ious <= threshold)[0]
        order = order[idxs + 1]

    return detections[keep, :]


def temporal_soft_nms(
    detections: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> np.ndarray:
    """Soft-NMS with Gaussian score decay for 1D temporal detections.

    Instead of hard suppression, reduces scores of overlapping proposals
    proportionally to their IoU. Preserves adjacent actions that hard NMS
    would eliminate.

    Args:
        detections: Array (n x 3) with columns [t_start, t_end, score].
        sigma: Gaussian decay width. Higher → softer penalization.
        score_threshold: Proposals with score below this are discarded.

    Returns:
        Filtered detections sorted by score descending.
    """
    dets = detections.copy()
    scores = dets[:, 2].copy()

    indices = list(range(len(dets)))
    keep = []

    while indices:
        # pick highest-score remaining detection
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
