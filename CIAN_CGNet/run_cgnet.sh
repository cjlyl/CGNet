#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash run_cgnet.sh DATA_DIRECTORY [r|s]" >&2
  exit 2
fi
variant=${2:-r}
case "$variant" in r|s) ;; *) echo "Expected variant r or s" >&2; exit 2 ;; esac
data="$(cd "$1" && pwd)"
python_bin=${PYTHON_BIN:-python}
embed=${EMBED_MODEL:-TransE}
for file in ent2ids relation2ids path_graph train_tasks_in_train.json dev_tasks.json \
            test_tasks.json rel2candidates_in_train.json e1rel_e2_in_train.json \
            "entity2vec.$embed" "relation2vec.$embed"; do
  if [[ ! -f "$data/$file" ]]; then
    echo "Missing dataset file: $data/$file" >&2
    exit 1
  fi
done
checkpoint=""
if [[ -n "${CIAN_CHECKPOINT:-}" ]]; then
  checkpoint="$(realpath -e "$CIAN_CHECKPOINT")"
fi
mkdir -p "${RUN_ROOT:-$project/runs}"
root="$(cd "${RUN_ROOT:-$project/runs}" && pwd)"
work="$(mktemp -d "$root/cian_${variant}_XXXXXX")"
cd "$work"

shared=(--device "${DEVICE:-cuda:0}" --datapath "$data" --few "${FEW:-5}"
        --seed "${SEED:-2022}" --embed_model "$embed" --max_neighbor "${MAX_NEIGHBOR:-100}")
prototype=(--cg_draws "${CG_DRAWS:-20}" --cg_temperature "${CG_TEMPERATURE:-1.0}")
if [[ -z "$checkpoint" ]]; then
  "$python_bin" -u "$project/main.py" "${shared[@]}" --fine_tune --lr "${CIAN_LR:-8e-5}" \
    --batch_size "${CIAN_BATCH_SIZE:-512}" --max_batches "${CIAN_STEPS:-300000}" \
    --warm_up_step "${CIAN_WARMUP:-10000}" --eval_every "${CIAN_EVAL_EVERY:-10000}" \
    --prefix backbone 2>&1 | tee backbone_train.log
  checkpoint="$work/models/backbone_best"
fi

"$python_bin" -u "$project/main.py" "${shared[@]}" --test --checkpoint "$checkpoint" \
  --prefix baseline 2>&1 | tee baseline_test.log
"$python_bin" -u "$project/main.py" "${shared[@]}" "${prototype[@]}" \
  --train_cgnet --checkpoint "$checkpoint" --cg_checkpoint cgnet.pt --cg_variant "$variant" \
  --cg_top_m "${CG_TOP_M:-500}" --cg_min_pairs "${CG_MIN_PAIRS:-10}" \
  --cg_steps "${CG_STEPS:-100000}" --cg_eval_every "${CG_EVAL_EVERY:-5000}" \
  --cg_batch_size "${CG_BATCH_SIZE:-64}" --prefix cgnet 2>&1 | tee cgnet_train.log
"$python_bin" -u "$project/main.py" "${shared[@]}" "${prototype[@]}" \
  --test --checkpoint "$checkpoint" --cg_checkpoint cgnet.pt \
  --prefix augmented 2>&1 | tee cgnet_test.log
echo "Checkpoints and logs: $work"
