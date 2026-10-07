#!/usr/bin/env bash
set -u
root="$1"
exec 9>"$root/restore.lock"
flock -x 9
if [ ! -f "$root/holder_stopped.json" ]; then
  printf '{"status":"NOT_STOPPED_NO_RESTORE_NEEDED"}\n' > "$root/gpu_holder_restore_status.json"
  exit 0
fi
script=/hpc/zhou228/GPU_computation/computation.py
python=/hpc/zhou228/GPU_computation/venv/bin/python
pattern="$script device=0 mem=50"
while true; do
  if ! pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 2>/dev/null)"; then
    printf '{"status":"WAITING_FOR_GPU_QUERY"}\n' > "$root/gpu_holder_restore_status.json"
    sleep 5;continue
  fi
  pending=0;holder_pid=""
  # A surviving child can still be loading data/model on CPU and allocate CUDA
  # later. Wait for that child, not just current nvidia-smi occupancy.
  if pgrep -u zhou228 -f "${root}/full50[.]py" >/dev/null;then pending=1;fi
  while IFS= read -r pid; do
    pid="$(echo "$pid" | tr -d '[:space:]')"
    [ -n "$pid" ] || continue
    cmd="$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)"
    case "$cmd" in
      *"$pattern"*) holder_pid="$pid" ;;
      *) pending=1 ;;
    esac
  done <<< "$pids"
  if [ "$pending" -eq 0 ];then break;fi
  printf '{"status":"WAITING_FOR_ALL_GPU0_TASKS"}\n' > "$root/gpu_holder_restore_status.json"
  sleep 5
done
state=ALREADY_RUNNING
if [ -z "$holder_pid" ]; then
  state=RESTORED
  nohup "$python" "$script" device=0 mem=50 9>&- >"$root/computation_restore.log" 2>&1 </dev/null &
  holder_pid="$!"
fi
ok=0
for i in $(seq 1 60);do
  if kill -0 "$holder_pid" 2>/dev/null && nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 2>/dev/null | tr -d ' ' | grep -qx "$holder_pid";then ok=1;break;fi
  sleep 2
done
if [ "$ok" -eq 0 ];then state=RESTORE_FAILED;fi
printf '{"status":"%s","pid":%s,"command":"%s device=0 mem=50","physical_gpu":0,"gpu_uuid":"GPU-bdf1c899-1408-3f77-05c1-4584e658333d"}\n' "$state" "$holder_pid" "$script" > "$root/gpu_holder_restore_status.json"
exit $((1-ok))
