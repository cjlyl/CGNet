#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash run_cgnet.sh DATA_DIRECTORY RECDAP_CHECKPOINT {r|s} [OPTIONS]" >&2
  exit 2
fi
project="$(cd "$(dirname "$0")" && pwd)"
data="$(cd "$1" && pwd)"
baseline="$(realpath -e "$2")"
variant="$3"
shift 3
case "$variant" in r|s) ;; *) echo "Variant must be r or s." >&2; exit 2 ;; esac
python_bin=${PYTHON_BIN:-python}
mkdir -p "${RUN_ROOT:-$project/runs}"
root="$(cd "${RUN_ROOT:-$project/runs}" && pwd)"
work="$(mktemp -d "$root/recdap_${variant}_XXXXXX")"

"$python_bin" -u "$project/main_cgnet.py" \
  --dataset "${DATASET:-NELL-One}" --data_path "$data" \
  --few "${FEW:-5}" --device "${DEVICE:-cpu}" \
  --baseline_checkpoint "$baseline" --cg_variant "$variant" \
  --step train --cg_checkpoint "$work/cgnet.pt" "$@" 2>&1 | tee "$work/run.log"
echo "Checkpoint and log: $work"
