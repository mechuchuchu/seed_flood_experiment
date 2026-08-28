#!/usr/bin/env bash
# Generates the full hyperparameter grid and fans it out to xargs -P N workers,
# each running run_single.py as a fresh, fully isolated OS process.
#
# Usage:
#   chmod +x run_parallel.sh
#   ./run_parallel.sh [n_parallel] [n_rounds]
#
# Requires: run_single.py and mnist_seedflood_sweep.py in the same directory.

set -euo pipefail

N_PARALLEL="${1:-$(nproc)}"
N_ROUNDS="${2:-300}"
OUTDIR="results_$(date +%Y%m%d_%H%M%S)"
COMBOFILE="$(mktemp)"

mkdir -p "$OUTDIR"
echo "outdir: $OUTDIR"
echo "parallelism: $N_PARALLEL"
echo "n_rounds per combo: $N_ROUNDS"

# ---- build combo list: mode lr mu beta1 beta2 ----
LRS=(1e-1 3e-2 1e-2 3e-3 1e-3 3e-4 1e-4 3e-5)
MUS=(1e-1 3e-2 1e-2 3e-3 1e-3)
BETAS=("0.9 0.999" "0.5 0.999" "0.9 0.99" "0.5 0.9" "0.0 0.999" "0.9 0.9999")

> "$COMBOFILE"

for lr in "${LRS[@]}"; do
  for mu in "${MUS[@]}"; do
    echo "sign $lr $mu 0.9 0.999" >> "$COMBOFILE"
    echo "raw  $lr $mu 0.9 0.999" >> "$COMBOFILE"
  done
done

for lr in "${LRS[@]}"; do
  for mu in "${MUS[@]}"; do
    for b in "${BETAS[@]}"; do
      echo "adam $lr $mu $b" >> "$COMBOFILE"
    done
  done
done

for lr in "${LRS[@]}"; do
  echo "fo $lr 1e-2 0.9 0.999" >> "$COMBOFILE"
done

N_COMBOS=$(wc -l < "$COMBOFILE")
echo "total combos: $N_COMBOS"

# ---- fan out via xargs -P ----
run_one() {
  mode="$1"; lr="$2"; mu="$3"; b1="$4"; b2="$5"
  tag="${mode}_lr${lr}_mu${mu}_b${b1}-${b2}"
  out="${OUTDIR}/${tag}.json"
  # timeout guard: 한 조합이 비정상적으로 오래 걸리면(무한루프/hang) 60초 후 강제 종료
  timeout 60s python3 run_single.py \
    --mode "$mode" --lr "$lr" --mu "$mu" --beta1 "$b1" --beta2 "$b2" \
    --n_rounds "$N_ROUNDS" --out "$out" \
    || echo "{\"mode\":\"$mode\",\"lr\":$lr,\"mu\":$mu,\"status\":\"TIMEOUT_OR_CRASH\"}" > "$out"
}
export -f run_one
export OUTDIR N_ROUNDS

cat "$COMBOFILE" | xargs -P "$N_PARALLEL" -L 1 bash -c 'run_one "$@"' _

rm -f "$COMBOFILE"

echo "done. aggregating..."
python3 aggregate_results.py "$OUTDIR"
