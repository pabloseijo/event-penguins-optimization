#!/usr/bin/env bash
# Valida o arranxo de add_rank_features sen repetir as 4 h de embeddings:
# apunta --train-repr/--val-repr aos .npz que xa calculou o intento que fallou.
set -u
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
C=BaseballPitch
F=$R/tmp/thumos14e_real/v1/eventpenguins_full/seed_1337/$C/local/fold_00
O=$R/tmp/proba_arranxo/$C
rm -rf "$O"; mkdir -p "$O"
eval "$(/usr/local/bin/gpu env --shell)"
echo "[$(date -Is)] probando con representacions xa calculadas"
env PYTHONPATH=.:dev OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$R/pyenv/bin/python" dev/train_quality_head.py \
    --data-path "$R/data/thumos14_real/preprocessed.h5" \
    --ann-path "$R/data/thumos14_real/config/annotations/by_class/$C/annotations.json" \
    --model-path "$R/models/model.pk" \
    --train-proposals "$F/lattice_train.csv" \
    --val-proposals "$F/lattice_val.csv" \
    --train-repr "$F/quality_head/cache/train_repr.npz" \
    --val-repr "$F/quality_head/cache/val_repr.npz" \
    --out-dir "$O" \
    --configs qhead_qfl_only \
    --epochs 18 --batch-size 4096 --repr-batch-size 16 --num-workers 2 \
    --max-train-samples 140000 \
    --max-train-proposals 50000 --max-val-proposals 20000 \
    --group-dro --group-dro-eta 0.01 --lr 1e-3 --weight-decay 1e-3 \
    --eval-every 6 --min-gt-duration 0.0 --min-score 0.1 \
    --pre-nms-topk-per-roi 0 --tiou 0.3 0.4 0.5 0.6 0.7 \
    --timestamp-cache-dir "$R/tmp/thumos14e_real/v1/shared_features/continuous/timestamp_cache" \
    --seed 1337 --device cuda:1
echo "[$(date -Is)] codigo de saida: $?"
