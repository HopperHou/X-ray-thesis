#!/usr/bin/env bash
set -u
root="$1";launcher="$2"
while kill -0 "$launcher" 2>/dev/null;do sleep 15;done
bash "$root/restore_holder.sh" "$root"
if [ ! -f "$root/completion_status.json" ];then
  printf '{"status":"INCOMPLETE","error":"launcher exited without final status"}\n' > "$root/completion_status.json"
fi
