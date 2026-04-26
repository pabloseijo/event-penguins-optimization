"""Temporal action detection evaluation: mAP and average recall.

Based on the ActivityNet evaluation toolkit. Supports optional joblib
parallelisation for multi-class scenarios.
"""

import json
import logging

import numpy as np
import pandas as pd
from absl import logging as absl_logging

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

logger = logging.getLogger(__name__)


class DetectionsEvaluator:
    GROUND_TRUTH_FIELDS = ["database", "version"]
    PREDICTION_FIELDS   = ["results",  "version"]

    def __init__(
        self,
        ground_truth_filename: str,
        prediction_filename: str,
        ground_truth_fields: list = GROUND_TRUTH_FIELDS,
        prediction_fields: list   = PREDICTION_FIELDS,
        tiou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
        verbose: bool = False,
        valid_sequences=None,
        valid_roi_ids=None,
        valid_labels=None,
        min_duration: float = 2.0,
    ):
        if not ground_truth_filename:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")

        self.tiou_thresholds = tiou_thresholds
        self.verbose         = verbose
        self.gt_fields       = ground_truth_fields
        self.pred_fields     = prediction_fields
        self.ap              = None

        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename, valid_sequences, valid_roi_ids, valid_labels, min_duration
        )
        self.prediction = self._import_prediction(
            prediction_filename, valid_sequences, valid_roi_ids, valid_labels, min_duration
        )

        if self.verbose:
            print(f"[INIT] Ground truth: {len(self.ground_truth)} instances.")
            print(f"[INIT] Predictions:  {len(self.prediction)} instances.")
            print(f"[INIT] tIoU thresholds: {self.tiou_thresholds}")

    def _import_ground_truth(
        self,
        ground_truth_filename: str,
        valid_sequences,
        valid_roi_ids,
        valid_labels,
        min_duration: float,
    ) -> tuple:
        """Load and validate the ground truth JSON file.

        Returns:
            ground_truth: DataFrame with columns [video-id, t-start, t-end, label].
            activity_index: Dict mapping label name → class index.
        """
        with open(ground_truth_filename, "r") as f:
            data = json.load(f)

        if not all(field in data for field in self.gt_fields):
            raise IOError("Please input a valid ground truth file.")

        activity_index: dict = {}
        cidx = 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []

        for videoid, v in data["database"].items():
            if valid_sequences is not None and videoid not in valid_sequences:
                continue
            for roi_id, roi_annotations in v["annotations"].items():
                if roi_id == "null":
                    continue
                if valid_roi_ids is not None and int(roi_id) not in valid_roi_ids:
                    continue
                for ann in roi_annotations:
                    if valid_labels is not None and ann["label"] not in valid_labels:
                        continue
                    if ann["segment"][1] - ann["segment"][0] < min_duration:
                        continue
                    if ann["label"] not in activity_index:
                        activity_index[ann["label"]] = cidx
                        cidx += 1
                    video_lst.append(f"{videoid}_{roi_id}")
                    t_start_lst.append(float(ann["segment"][0]))
                    t_end_lst.append(float(ann["segment"][1]))
                    label_lst.append(activity_index[ann["label"]])

        ground_truth = pd.DataFrame({
            "video-id": video_lst,
            "t-start":  t_start_lst,
            "t-end":    t_end_lst,
            "label":    label_lst,
        })
        return ground_truth, activity_index

    def _import_prediction(
        self,
        prediction_filename: str,
        valid_sequences,
        valid_roi_ids,
        valid_labels,
        min_duration: float,
    ) -> pd.DataFrame:
        """Load and validate the prediction JSON file.

        Returns:
            DataFrame with columns [video-id, t-start, t-end, label, score].
        """
        with open(prediction_filename, "r") as f:
            data = json.load(f)

        if not all(field in data for field in self.pred_fields):
            raise IOError("Please input a valid prediction file.")

        video_lst, t_start_lst, t_end_lst, label_lst, score_lst = [], [], [], [], []

        for videoid, v in data["results"].items():
            if valid_sequences is not None and videoid not in valid_sequences:
                continue
            for roi_id, roi_annotation in v.items():
                if roi_id == "null":
                    continue
                if valid_roi_ids is not None and int(roi_id) not in valid_roi_ids:
                    continue
                for result in roi_annotation:
                    if valid_labels is not None and result["label"] not in valid_labels:
                        continue
                    if result["segment"][1] - result["segment"][0] < min_duration:
                        continue
                    video_lst.append(f"{videoid}_{roi_id}")
                    t_start_lst.append(float(result["segment"][0]))
                    t_end_lst.append(float(result["segment"][1]))
                    label_lst.append(self.activity_index[result["label"]])
                    score_lst.append(result["score"])

        return pd.DataFrame({
            "video-id": video_lst,
            "t-start":  t_start_lst,
            "t-end":    t_end_lst,
            "label":    label_lst,
            "score":    score_lst,
        })

    def _get_predictions_with_label(
        self, prediction_by_label, label_name: str, cidx: int
    ) -> pd.DataFrame:
        """Return predictions for a given class, or an empty DataFrame if none exist."""
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except KeyError:
            absl_logging.warning(f"No predictions for label '{label_name}'.")
            return pd.DataFrame()

    def wrapper_compute_average_precision(self) -> np.ndarray:
        """Compute per-class AP across all tIoU thresholds.

        Returns:
            Array of shape (num_tiou, num_classes).
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        gt_by_label   = self.ground_truth.groupby("label")
        pred_by_label = self.prediction.groupby("label")

        if _JOBLIB:
            results = Parallel(n_jobs=len(self.activity_index))(
                delayed(compute_average_precision_detection)(
                    ground_truth=gt_by_label.get_group(cidx).reset_index(drop=True),
                    prediction=self._get_predictions_with_label(pred_by_label, name, cidx),
                    tiou_thresholds=self.tiou_thresholds,
                )
                for name, cidx in self.activity_index.items()
            )
        else:
            results = [
                compute_average_precision_detection(
                    ground_truth=gt_by_label.get_group(cidx).reset_index(drop=True),
                    prediction=self._get_predictions_with_label(pred_by_label, name, cidx),
                    tiou_thresholds=self.tiou_thresholds,
                )
                for name, cidx in self.activity_index.items()
            ]

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]
        return ap

    def wrapper_compute_average_recall(self, max_pred=None) -> np.ndarray:
        """Compute per-class AR across all tIoU thresholds.

        Returns:
            Array of shape (num_tiou, num_classes).
        """
        ar = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        gt_by_label   = self.ground_truth.groupby("label")
        pred_by_label = self.prediction.groupby("label")

        if _JOBLIB:
            results = Parallel(n_jobs=len(self.activity_index))(
                delayed(compute_average_recall)(
                    ground_truth=gt_by_label.get_group(cidx).reset_index(drop=True),
                    prediction=self._get_predictions_with_label(pred_by_label, name, cidx),
                    tiou_thresholds=self.tiou_thresholds,
                    max_pred=max_pred,
                )
                for name, cidx in self.activity_index.items()
            )
        else:
            results = [
                compute_average_recall(
                    ground_truth=gt_by_label.get_group(cidx).reset_index(drop=True),
                    prediction=self._get_predictions_with_label(pred_by_label, name, cidx),
                    tiou_thresholds=self.tiou_thresholds,
                    max_pred=max_pred,
                )
                for name, cidx in self.activity_index.items()
            ]

        for i, cidx in enumerate(self.activity_index.values()):
            ar[:, cidx] = results[i]
        return ar

    def run(self) -> float:
        """Compute interpolated mAP and return the average across tIoU thresholds."""
        self.ap        = self.wrapper_compute_average_precision()
        self.mAP       = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print(f"[RESULTS] Average-mAP: {self.average_mAP:.4f}")
        return self.average_mAP

    def evaluate_recall(self, max_pred=None) -> pd.DataFrame:
        """Compute AR and return as a DataFrame indexed by tIoU threshold."""
        self.ar = self.wrapper_compute_average_recall(max_pred=max_pred)
        labels  = list(self.activity_index)
        ar_df   = pd.DataFrame(self.ar, columns=labels)
        ar_df.insert(0, "tiou_threshold", self.tiou_thresholds)
        return ar_df


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------

def segment_iou(target_segment: np.ndarray, candidate_segments: np.ndarray) -> np.ndarray:
    """Compute temporal IoU between one target segment and N candidates.

    Args:
        target_segment: [t_start, t_end], shape (2,).
        candidate_segments: Array of shape (N, 2) with [t_start, t_end] rows.

    Returns:
        Array of shape (N,) with IoU scores.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    intersection = (tt2 - tt1).clip(0)
    union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0])
        + (target_segment[1] - target_segment[0])
        - intersection
    )
    return intersection.astype(float) / union


def interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> float:
    """Interpolated AP — VOCdevkit from VOC 2011.

    Args:
        prec: Precision array.
        rec: Recall array.

    Returns:
        Scalar AP value.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec  = np.hstack([[0], rec,  [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def compute_average_precision_detection(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    tiou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
) -> np.ndarray:
    """Compute AP for one class across multiple tIoU thresholds.

    Only the highest-scoring prediction is counted as a true positive per
    ground-truth segment. Inspired by Pascal VOC devkit.

    Args:
        ground_truth: DataFrame with columns [video-id, t-start, t-end].
        prediction: DataFrame with columns [video-id, t-start, t-end, score].
        tiou_thresholds: Array of IoU thresholds to evaluate at.

    Returns:
        Array of shape (len(tiou_thresholds),) with AP per threshold.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        absl_logging.warning("Evaluator returned 0 due to empty predictions!")
        return ap

    npos    = float(len(ground_truth))
    lock_gt = np.full((len(tiou_thresholds), len(ground_truth)), -1)

    sort_idx   = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    gt_by_video = ground_truth.groupby("video-id")

    for idx, this_pred in prediction.iterrows():
        try:
            gt_video = gt_by_video.get_group(this_pred["video-id"])
        except KeyError:
            fp[:, idx] = 1
            continue

        this_gt   = gt_video.reset_index()
        tiou_arr  = segment_iou(
            this_pred[["t-start", "t-end"]].values,
            this_gt[["t-start", "t-end"]].values,
        )
        tiou_sorted_idx = tiou_arr.argsort()[::-1]

        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break
            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum  = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum  = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum    = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx], recall_cumsum[tidx])
    return ap


def compute_average_recall(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    tiou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
    max_pred: int = None,
) -> np.ndarray:
    """Compute AR for one class across multiple tIoU thresholds.

    Args:
        ground_truth: DataFrame with columns [video-id, t-start, t-end].
        prediction: DataFrame with columns [video-id, t-start, t-end, score].
        tiou_thresholds: Array of IoU thresholds to evaluate at.
        max_pred: Maximum number of predictions per video (None = all).

    Returns:
        Array of shape (len(tiou_thresholds),) with recall per threshold.
    """
    if prediction.empty:
        absl_logging.warning("Evaluator returned 0 due to empty predictions!")
        return np.zeros(len(tiou_thresholds))

    video_ids = ground_truth["video-id"].unique()
    filtered = pd.concat(
        [
            prediction[prediction["video-id"] == vid]
            .sort_values("score", ascending=False)
            .head(max_pred)
            for vid in video_ids
        ],
        ignore_index=True,
    )

    pred_by_video = filtered.groupby("video-id")
    tp = np.zeros(len(tiou_thresholds), dtype=int)

    for _, this_gt in ground_truth.iterrows():
        vid = this_gt["video-id"]
        if vid not in pred_by_video.groups:
            logger.warning(f"No predictions for sequence {vid}. Skipping.")
            continue
        this_pred = pred_by_video.get_group(vid).reset_index(drop=True)
        tiou_arr  = segment_iou(
            this_gt[["t-start", "t-end"]].values,
            this_pred[["t-start", "t-end"]].values,
        )
        tp += tiou_thresholds <= np.max(tiou_arr)

    return tp / len(ground_truth)
