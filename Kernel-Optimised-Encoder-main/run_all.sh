#!/usr/bin/env bash
# PS-09 GPU workflow for the Ubuntu RTX machine (also works in WSL / Git Bash with an NVIDIA GPU).
#
#   bash run_all.sh                 quick path: env + accuracy + ablation on 20k + budget sweep + latency + profiles
#   bash run_all.sh --full          also: full 100k official bench of the submitted config + full 100k accuracy check
#   bash run_all.sh --step NAME     run one step only (env, test, validate, ablation, sweep, latency, profile, full, collect)
#
# Restartable: a step whose result file already exists is skipped (delete the file to redo it).
# The 1-hour FP32 reference over 100k passages is NEVER run: reference.py pads every input to 512 tokens, so its time
# per passage is independent of the text and the 5,000-passage sample in step "env" is the baseline (see README).
# All results go to results_gpu/<category>/ ; nothing is marked passed unless its script exited 0.
set -uo pipefail
cd "$(dirname "$0")"
FULL=0; ONLY=""
while [ $# -gt 0 ]; do case "$1" in --full) FULL=1;; --step) ONLY="$2"; shift;; *) echo "unknown arg $1"; exit 2;; esac; shift; done
R=results_gpu; mkdir -p $R/{hardware,accuracy,ablation,token_budget,latency,profiler,final}
STATUS=$R/STATUS.txt; touch $STATUS
log() { echo "[$(date '+%F %T')] $*" | tee -a $STATUS; }
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
run() { local name="$1"; shift; log "START $name"; if "$@"; then log "OK    $name"; else log "FAIL  $name (exit $?)"; fi; }

# ---------------------------------------------------------------- python env (project-local, no system changes)
PYBIN=$(command -v python3 || command -v python)
[ -d .venv ] || "$PYBIN" -m venv .venv
if [ -f .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
if ! "$PY" -c "import torch, transformers, numpy" 2>/dev/null; then
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements-gpu.txt
fi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available: install a CUDA build of torch (requirements-gpu.txt)'" || exit 1
[ -f data/practice_passages.jsonl ] || "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('Tarakeshwaran/CIT_Hackathon_PS09_encoder', repo_type='dataset', local_dir='data')"

# ---------------------------------------------------------------- 1. hardware + baseline
if want env; then
  nvidia-smi > $R/hardware/nvidia_smi.txt 2>&1
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,clocks.max.sm,power.limit --format=csv > $R/hardware/gpu.csv 2>&1
  "$PY" scripts/env_report.py > $R/hardware/env.json
  lscpu > $R/hardware/lscpu.txt 2>/dev/null || true
  # recover the earlier baseline if it exists in the old location
  [ -s $R/hardware/ref_throughput.json ] || { [ -s results/ref_throughput.json ] && cp results/ref_throughput.json $R/hardware/; }
  if [ ! -s $R/hardware/ref_throughput.json ]; then
    run "reference baseline (5,000 passages, FP32 pad 512)" bash -c \
      "$PY data/reference.py --passages data/practice_passages.jsonl --out $R/hardware/ref5k.npy --limit 5000 2>$R/hardware/ref_baseline.log | tail -1 > $R/hardware/ref_throughput.json"
  fi
  cat $R/hardware/ref_throughput.json
fi

# ---------------------------------------------------------------- 2. correctness: edge cases on GPU (FP16, graphs)
if want test && [ ! -s $R/accuracy/test_correctness.json ]; then
  run "edge-case correctness, stages 1-4" "$PY" tests/test_correctness.py --data data --stages 1 2 3 4 --out $R/accuracy/test_correctness.json
fi

# ---------------------------------------------------------------- 3. accuracy gate on the official references
if want validate; then
  [ -s $R/accuracy/validate_default.json ] || run "validate submitted default (2,000 passages + 1,500 queries)" \
    env PS09_DEBUG=1 "$PY" validate.py --data data --out $R/accuracy/validate_default.json
  [ -s $R/accuracy/validate_fp16_plain.json ] || run "validate plain FP16 residual stream (PS09_RES32=0)"     env PS09_RES32=0 PS09_SHORT_FP32=0 PS09_DEBUG=1 "$PY" validate.py --data data --out $R/accuracy/validate_fp16_plain.json
  [ -s $R/accuracy/validate_bf16.json ] || run "validate BF16 variant (experiment)" \
    env PS09_BF16=1 "$PY" validate.py --data data --out $R/accuracy/validate_bf16.json
  [ -s $R/accuracy/validate_lax_reduction.json ] || run "validate with cuBLAS reduced-precision reductions allowed (experiment)" \
    env PS09_STRICT_REDUCTION=0 "$PY" validate.py --data data --out $R/accuracy/validate_lax_reduction.json
  [ -s $R/accuracy/validate_fp16_accum.json ] || run "validate FP16 accumulation (experiment, expected to fail)" \
    env PS09_FP16_ACCUM=1 "$PY" validate.py --data data --out $R/accuracy/validate_fp16_accum.json
fi

# ---------------------------------------------------------------- 4. ablation with the official bench.py (20k passages)
if want ablation && [ ! -s $R/ablation/ABLATION.md ]; then
  # S1 still pads to 512 like the reference: ~5 min on 20k on a laptop GPU. Live trace only where graphs exist.
  run "ablation S1-S2 (20k, official bench.py)" "$PY" run_ablation.py --data data --stages 1 2 --limit 20000 --no-trace --out-dir $R/ablation --tag "ablation 20k"
  run "ablation S3-S5 (20k + live trace, official bench.py)" "$PY" run_ablation.py --data data --stages 3 4 5 --limit 20000 --out-dir $R/ablation --tag "ablation 20k + live"
  run "S4 with plain FP16 residual stream (RES32 off)" env PS09_RES32=0 "$PY" run_ablation.py --data data --stages 4 --limit 20000 --no-trace --out-dir $R/ablation --tag "RES32 off"
  run "S4 without length-bucket split (O2b off)" env PS09_BUCKET=0 "$PY" run_ablation.py --data data --stages 4 --limit 20000 --no-trace --out-dir $R/ablation --tag "O2b off"
fi

# ---------------------------------------------------------------- 5. token-budget sweep (stage 4, 20k passages)
if want sweep && [ ! -s $R/token_budget/ABLATION.md ]; then
  for B in 8192 16384 32768 65536 131072; do
    run "budget $B" env PS09_TOKEN_BUDGET=$B "$PY" run_ablation.py --data data --stages 4 --limit 20000 --no-trace --out-dir $R/token_budget --tag "budget sweep"
  done
  "$PY" scripts/apply_best_config.py --sweep $R/token_budget/ablation.csv | tee $R/token_budget/recommendation.txt
fi

# ---------------------------------------------------------------- 6. live-query latency (non-official detail; bench.py gives the official p50/p99)
if want latency; then
  [ -s $R/latency/service_graph_on.json ]  || run "service time, CUDA Graph on"  "$PY" scripts/live_latency.py --data data --mode service --out $R/latency/service_graph_on.json
  [ -s $R/latency/service_graph_off.json ] || run "service time, CUDA Graph off" env PS09_GRAPH=0 "$PY" scripts/live_latency.py --data data --mode service --out $R/latency/service_graph_off.json
  [ -s $R/latency/realtime_default.json ]  || run "real-time trace replay (~4 min)" "$PY" scripts/live_latency.py --data data --out $R/latency/realtime_default.json
fi

# ---------------------------------------------------------------- 7. profiler evidence (before / after)
if want profile; then
  [ -s $R/profiler/reference/summary.json ] || run "profile reference (2,000 passages)" "$PY" profile_run.py --which reference --data data --n 2000 --out $R/profiler/reference
  [ -s $R/profiler/stage1/summary.json ]    || run "profile stage 1" env PS09_STAGE=1 "$PY" profile_run.py --which team --data data --n 2000 --out $R/profiler/stage1
  [ -s $R/profiler/stage2/summary.json ]    || run "profile stage 2" env PS09_STAGE=2 "$PY" profile_run.py --which team --data data --n 2000 --out $R/profiler/stage2
  [ -s $R/profiler/default/summary.json ]   || run "profile submitted default" "$PY" profile_run.py --which team --data data --n 2000 --out $R/profiler/default
  if command -v nsys >/dev/null; then
    [ -s $R/profiler/default_nsys.nsys-rep ] || run "Nsight Systems timeline (default, 20k)" \
      nsys profile -o $R/profiler/default_nsys --force-overwrite true -t cuda,nvtx,osrt "$PY" scripts/full_check.py run --data data --limit 20000 --out $R/profiler/nsys_tmp.npy
  fi
fi

# ---------------------------------------------------------------- 8. full 100k runs (explicit: --full or --step full)
if [ $FULL = 1 ] || [ "$ONLY" = "full" ]; then
  [ -s $R/final/ABLATION.md ] || run "FINAL official bench.py, submitted default, 100k + live trace" \
    "$PY" run_ablation.py --data data --stages $("$PY" -c "import re;print(re.search(r'^DEFAULT_STAGE = (\d+)', open('team.py').read(), re.M).group(1))") --out-dir $R/final --tag "final 100k"
  [ -s $R/final/submitted_100k.json ] || run "full 100k embeddings, submitted default" "$PY" scripts/full_check.py run --data data --out $R/final/submitted_100k.npy
  [ -s $R/final/fp32_100k.json ] || run "full 100k embeddings, FP32 path (reference maths)" env PS09_STAGE=2 PS09_FP16=0 "$PY" scripts/full_check.py run --data data --out $R/final/fp32_100k.npy
  [ -s $R/final/fp16plain_100k.json ] || run "full 100k embeddings, plain FP16 (RES32 off, SHORT_FP32 off)" env PS09_RES32=0 PS09_SHORT_FP32=0 "$PY" scripts/full_check.py run --data data --out $R/final/fp16plain_100k.npy
  [ -s $R/final/full_check_fp16plain.json ] || run "compare 100k: plain FP16 vs FP32" "$PY" scripts/full_check.py compare --data data --a $R/final/fp16plain_100k.npy --b $R/final/fp32_100k.npy --out $R/final/full_check_fp16plain.json
  [ -s $R/final/full_check.json ] || run "compare 100k: submitted vs FP32" "$PY" scripts/full_check.py compare --data data --a $R/final/submitted_100k.npy --b $R/final/fp32_100k.npy --out $R/final/full_check.json
fi

# ---------------------------------------------------------------- 9. summary
if want collect || [ -z "$ONLY" ]; then
  "$PY" scripts/collect_results.py --results $R | tee $R/SUMMARY.md
fi
log "done. Copy the whole results_gpu/ folder back (it is small; *.npy files are git-ignored)."
