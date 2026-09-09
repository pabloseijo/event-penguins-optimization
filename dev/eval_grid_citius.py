import json, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path("/home/pablo.garcia.seijo/event_penguins")))
from src.evaluation import DetectionsEvaluator
from src.utils import temporal_soft_nms

ANN = "/home/pablo.garcia.seijo/event_penguins/config/annotations/annotations.json"
SCORES_CSV = "/home/pablo.garcia.seijo/event_penguins/tmp/quality_head/lattice_variant_features_qonly_test_top1000_grid/cache/test_lattice_top2000_cnn_dur_variant_features_qonly_top1000_grid_scores_qhead_qfl_only.csv"
OUT_DIR = Path("/home/pablo.garcia.seijo/event_penguins/tmp/quality_head/lattice_variant_features_qonly_test_top1000_grid")
TIOU = [0.1, 0.3, 0.5, 0.7]

df = pd.read_csv(SCORES_CSV)
print(f"Proposals: {len(df)}, recs: {df.rec_name.nunique()}", flush=True)

def make_preds(df, score_col, min_score=0.05, sigma=0.25, use_boundary=True):
    result = {}
    for (rec, roi), grp in df.groupby(["rec_name", "roi_id"]):
        roi_key = str(int(roi.lstrip("N"))) if isinstance(roi, str) else str(roi)
        scores = grp[score_col].to_numpy(dtype=np.float64)
        if use_boundary and "refined_t_start" in grp.columns:
            t_start = grp["refined_t_start"].to_numpy()
            t_end = grp["refined_t_end"].to_numpy()
            bad = (t_end <= t_start) | (t_start < 0)
            t_start = np.where(bad, grp["t_start"].to_numpy()/1e6, t_start)
            t_end = np.where(bad, grp["t_end"].to_numpy()/1e6, t_end)
        else:
            t_start = grp["t_start"].to_numpy()/1e6
            t_end = grp["t_end"].to_numpy()/1e6
        mask = scores >= min_score
        if not mask.any():
            result.setdefault(rec, {})[roi_key] = []
            continue
        arr = np.column_stack([t_start[mask], t_end[mask], scores[mask]])
        kept = temporal_soft_nms(arr, sigma=sigma, score_threshold=0.001)
        result.setdefault(rec, {})[roi_key] = [
            {"label":"ed","segment":[float(r[0]),float(r[1])],"score":float(r[2])}
            for r in kept if r[1]-r[0] >= 2.0
        ]
    return {"version":"grid_test","results":result}

rows = []
score_cols = ["cnn_score", "quality_score", "quality_avg_cnn", "sqrt_quality_x_cnn", "quality_score_base_priority_080", "quality_avg_cnn_base_priority_080"]
for sc in score_cols:
    if sc not in df.columns: continue
    for ms in [0.01, 0.05, 0.1, 0.2]:
        for boundary in [True, False]:
            p = make_preds(df, sc, ms, use_boundary=boundary)
            tmp = OUT_DIR/"_tmp.json"; tmp.write_text(json.dumps(p))
            ev = DetectionsEvaluator(
                ground_truth_filename=ANN,
                prediction_filename=str(tmp),
                tiou_thresholds=np.array(TIOU),
                valid_labels="ed",
                valid_sequences=list(p["results"].keys()),
                min_duration=2.0,
            )
            mAP = ev.run()
            n = sum(len(v) for rois in p["results"].values() for v in rois.values())
            row = {"score_col":sc,"min_score":ms,"boundary":boundary,"mAP":round(mAP,6),"n_pred":n}
            for i,t in enumerate(TIOU): row[f"AP@{t}"] = round(float(ev.mAP[i]),6)
            rows.append(row)
            print(f"{sc:42s} ms={ms:.2f} bnd={boundary}: mAP={mAP:.4f} AP@0.7={ev.mAP[3]:.4f} n={n}", flush=True)

out = pd.DataFrame(rows).sort_values("mAP",ascending=False)
out.to_csv(OUT_DIR/"summary_grid_eval.csv",index=False)
print("\nTop 10:")
print(out.head(10).to_string(index=False))
