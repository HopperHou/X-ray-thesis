#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
exec 8>"$root/pipeline.lock"
flock -n 8 || exit 1
export CUDA_VISIBLE_DEVICES=GPU-bdf1c899-1408-3f77-05c1-4584e658333d
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
py=/hpc/zhou228/venvs/nnunet/bin/python
child=""
cleanup() {
  code=$?
  trap - EXIT TERM INT
  if [ -n "$child" ] && kill -0 "$child" 2>/dev/null;then
    kill -TERM -- "-$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  if [ "$code" -ne 0 ] && [ ! -f "$root/completion_status.json" ];then
    "$py" "$root/full50.py" invalid --error "pipeline exited code $code" || true
  fi
  bash "$root/restore_holder.sh" "$root" || true
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
nohup bash "$root/restore_guard.sh" "$root" "$$" 8>&- >"$root/restore_guard.log" 2>&1 </dev/null &
run() {
  setsid "$py" "$@" 8>&- &
  child=$!
  wait "$child"
  child=""
}
run "$root/full50.py" p0
run "$root/full50.py" p1
# Check the restore command exists before releasing any GPU memory.
test -f /hpc/zhou228/GPU_computation/computation.py
test -x /hpc/zhou228/GPU_computation/venv/bin/python
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0);do
  owner="$(ps -p "$pid" -o user= | tr -d '[:space:]')"
  cmd="$(tr '\0' ' ' </proc/"$pid"/cmdline)"
  case "$cmd" in
    *'/hpc/zhou228/GPU_computation/computation.py device=0 '*)
      test "$owner" = zhou228
      printf '{"holder_pid":%s,"stopped_for":"AXIS_FULL50_PLAN_B"}\n' "$pid" > "$root/holder_stopped.json"
      kill -TERM "$pid"
      for i in $(seq 1 30);do kill -0 "$pid" 2>/dev/null || break;sleep 1;done
      if kill -0 "$pid" 2>/dev/null;then kill -KILL "$pid";fi
      ;;
    *) echo "Resource conflict: GPU0 process $pid is not the authorized holder";exit 1 ;;
  esac
done
if [ ! -f "$root/holder_stopped.json" ];then
  printf '{"holder_pid":null,"already_stopped":true}\n' > "$root/holder_stopped.json"
fi
sleep 2
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0)"
run "$root/full50.py" resource
for seed in 42 43 44;do
  for arm in B2 AXIS;do
    echo "Starting seed${seed}/${arm} E1-E50"
    run "$root/full50.py" train --seed "$seed" --arm "$arm"
  done
done
run "$root/full50.py" pair
run "$root/full50.py" eval
# All GPU work is now finished; restore immediately, before CPU bootstrap.
bash "$root/restore_holder.sh" "$root"
run "$root/full50_statistics.py"
echo "Pipeline complete; see completion_status.json and REPORT.md"
