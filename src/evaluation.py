"""Detection metrics: ActivityNet-style mAP and average recall.

The evaluator reads an ActivityNet-style ground truth and prediction file and
reports mean average precision over a set of tIoU thresholds, plus average recall
for proposal-level analysis. The scope of an evaluation can be narrowed to a set
of recordings, ROIs or labels, which is what makes per-fold and per-ROI numbers
comparable with the full-split ones.
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
    """Score a prediction file against the ground truth over several tIoU values.

    Args:
        ground_truth_filename: ActivityNet-style annotation file.
        prediction_filename: prediction file written by the classification stage.
        ground_truth_fields: required top-level fields of the annotation file.
        prediction_fields: required top-level fields of the prediction file.
        tiou_thresholds: tIoU thresholds averaged into the reported mAP.
        verbose: print the resulting average mAP.
        valid_sequences: restrict scoring to these recordings.
        valid_roi_ids: restrict scoring to these ROIs.
        valid_labels: restrict scoring to these labels.
        min_duration: ground-truth instances shorter than this are ignored.

    Raises:
        IOError: if either filename is empty or cannot be read.
    """
    GROUND_TRUTH_FIELDS = ["database", "version"]
    PREDICTION_FIELDS = ["results", "version"]

    def __init__(
        self,
        ground_truth_filename: str,
        prediction_filename: str,
        ground_truth_fields: list = GROUND_TRUTH_FIELDS,
        prediction_fields: list = PREDICTION_FIELDS,
        tiou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
        verbose: bool = False,
        valid_sequences=None,
        valid_roi_ids=None,
        valid_labels=None,
        min_duration: float = 2.0,
    ):
        if not ground_truth_filename:
            raise IOError("Introduce un ficheiro de ground truth válido.")
        if not prediction_filename:
            raise IOError("Introduce un ficheiro de predicións válido.")

        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None

        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename, valid_sequences, valid_roi_ids, valid_labels, min_duration
        )
        self.prediction = self._import_prediction(
            prediction_filename, valid_sequences, valid_roi_ids, valid_labels, min_duration
        )

        if self.verbose:
            print(f"[INIT] Anotacións cargadas: {len(self.ground_truth)} instancias.")
            print(f"[INIT] Predicións cargadas: {len(self.prediction)} instancias.")
            print(f"[INIT] Umbrais tIoU: {self.tiou_thresholds}")

    def _import_ground_truth(self, ground_truth_filename, valid_sequences,
                              valid_roi_ids, valid_labels, min_duration):
        with open(ground_truth_filename, "r") as f:
            data = json.load(f)

        if not all(field in data for field in self.gt_fields):
            raise IOError("Ficheiro de ground truth non válido.")

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

        return pd.DataFrame({
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
        }), activity_index

    def _import_prediction(self, prediction_filename, valid_sequences,
                            valid_roi_ids, valid_labels, min_duration):
        with open(prediction_filename, "r") as f:
            data = json.load(f)

        if not all(field in data for field in self.pred_fields):
            raise IOError("Ficheiro de predicións non válido.")

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
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        })

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except KeyError:
            absl_logging.warning(f"Sen predicións para a clase '{label_name}'.")
            return pd.DataFrame()

    def wrapper_compute_average_precision(self) -> np.ndarray:
        """Compute average precision per label and tIoU threshold, in parallel.

        Returns:
            ``[len(tiou_thresholds), num_labels]`` array of average precisions.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        gt_by_label = self.ground_truth.groupby("label")
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
        """Compute average recall per label and tIoU threshold, in parallel.

        Args:
            max_pred: keep only the top-scoring predictions per recording, i.e. the
                ``@k`` of AR@k. None keeps all of them.

        Returns:
            ``[len(tiou_thresholds), num_labels]`` array of average recalls.
        """
        ar = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))
        gt_by_label = self.ground_truth.groupby("label")
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
        """Return the average mAP over the configured tIoU thresholds.

        The per-threshold mAP stays available as ``self.mAP`` and the per-label average
        precision as ``self.ap``.
        """
        self.ap = self.wrapper_compute_average_precision()
        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print(f"[RESULTADOS] Average-mAP: {self.average_mAP:.4f}")
        return self.average_mAP

    def evaluate_recall(self, max_pred=None) -> pd.DataFrame:
        """Return average recall as a DataFrame indexed by tIoU threshold.

        Args:
            max_pred: number of top-scoring predictions kept per recording.

        Returns:
            DataFrame with one ``tiou_threshold`` column and one column per label.
        """
        self.ar = self.wrapper_compute_average_recall(max_pred=max_pred)
        labels = list(self.activity_index)
        ar_df = pd.DataFrame(self.ar, columns=labels)
        ar_df.insert(0, "tiou_threshold", self.tiou_thresholds)
        return ar_df


def segment_iou(target_segment: np.ndarray, candidate_segments: np.ndarray) -> np.ndarray:
    """Temporal IoU between one segment and an array of candidate segments.

    Args:
        target_segment: ``[t_start, t_end]`` of the reference segment.
        candidate_segments: ``[n, 2]`` array of candidate segments.

    Returns:
        ``[n]`` array of tIoU values.
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
    """AP interpolada — VOCdevkit de VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def compute_average_precision_detection(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    tiou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
) -> np.ndarray:
    """Average precision of one label, following the ActivityNet protocol.

    Predictions are ranked by score; a prediction is a true positive when it exceeds
    the tIoU threshold against a ground-truth instance of the same recording that no
    higher-scoring prediction has already claimed.

    Args:
        ground_truth: instances of this label, with ``video-id``, ``t-start``, ``t-end``.
        prediction: predictions of this label, with an extra ``score`` column.
        tiou_thresholds: thresholds to evaluate.

    Returns:
        ``[len(tiou_thresholds)]`` array of average precisions; zeros when there is
        no prediction.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        absl_logging.warning("Avaliador devolveu 0 por predicións baleiras.")
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.full((len(tiou_thresholds), len(ground_truth)), -1)
    sort_idx = prediction["score"].values.argsort()[::-1]
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

        this_gt = gt_video.reset_index()
        tiou_arr = segment_iou(
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

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos
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
    """Fraction of ground-truth instances recovered by the top predictions.

    An instance counts as recovered when any kept prediction of its recording reaches
    the tIoU threshold, regardless of rank. This is the proposal-level metric used to
    compare proposal generators independently of the classifier.

    Args:
        ground_truth: instances of this label.
        prediction: predictions of this label.
        tiou_thresholds: thresholds to evaluate.
        max_pred: predictions kept per recording, i.e. the ``@k`` of AR@k.

    Returns:
        ``[len(tiou_thresholds)]`` array of recall values.
    """
    if prediction.empty:
        absl_logging.warning("Avaliador devolveu 0 por predicións baleiras.")
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
            logger.warning(f"Sen predicións para {vid}. Omitindo.")
            continue
        this_pred = pred_by_video.get_group(vid).reset_index(drop=True)
        tiou_arr = segment_iou(
            this_gt[["t-start", "t-end"]].values,
            this_pred[["t-start", "t-end"]].values,
        )
        tp += tiou_thresholds <= np.max(tiou_arr)

    return tp / len(ground_truth)
