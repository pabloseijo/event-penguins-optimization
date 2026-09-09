#!/usr/bin/env bash
# Mide canto custa de verdade a quality head por proposta.
#
# Fai falla porque a etapa non da ningunha pista: escribe as representacions nun
# unico .npz ao final e o pipeline pasalle --quiet-progress. O 2026-09-07 estivo
# 5 h cun lattice de 4,08 M propostas sen producir un so byte.
#
# A causa atopouse aqui: ProposalDataset ten cache_full_events=True por defecto e
# train_quality_head.py construiao sen desactivalo, asi que materializaba a
# gravacion ENTEIRA por cada ROI. Tres workers ocupaban 7,4 + 8,6 + 10,9 GB.
# Esta calibracion mide o antes/despois dese arranxo.
set -u
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
C=${C:-BaseballPitch}
NTR=${NTR:-20000}
NVA=${NVA:-8000}
F=$R/tmp/thumos14e_real/v1/eventpenguins_full/seed_1337/$C/local/fold_00
O=$R/tmp/calibracion_qhead/$C
TC=$R/tmp/thumos14e_real/v1/shared_features/continuous/timestamp_cache
rm -rf "$O"; mkdir -p "$O"

eval "$(/usr/local/bin/gpu env --shell)"
echo "[$(date -Is)] calibrando $C · $NTR train / $NVA val"
t0=$(date +%s)
env PYTHONPATH=.:dev OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$R/pyenv/bin/python" dev/train_quality_head.py \
    --data-path "$R/data/thumos14_real/preprocessed.h5" \
    --ann-path "$R/data/thumos14_real/config/annotations/by_class/$C/annotations.json" \
    --model-path "$R/models/model.pk" \
    --train-proposals "$F/lattice_train.csv" \
    --val-proposals "$F/lattice_val.csv" \
    --out-dir "$O" \
    --configs qhead_qfl_only \
    --epochs 18 --batch-size 4096 --repr-batch-size 16 --num-workers 3 \
    --max-train-samples 140000 \
    --max-train-proposals "$NTR" --max-val-proposals "$NVA" \
    --group-dro --group-dro-eta 0.01 --lr 1e-3 --weight-decay 1e-3 \
    --eval-every 6 --min-gt-duration 0.0 --min-score 0.1 \
    --pre-nms-topk-per-roi 0 \
    --tiou 0.3 0.4 0.5 0.6 0.7 \
    --timestamp-cache-dir "$TC" \
    --seed 1337 --device cuda:1
rc=$?
t1=$(date +%s); d=$((t1 - t0)); n=$((NTR + NVA))
echo "[$(date -Is)] rematou con codigo $rc en ${d}s para $n propostas"
[ "$d" -gt 0 ] && echo "  ritmo global: $(( n * 3600 / d )) propostas/h"
