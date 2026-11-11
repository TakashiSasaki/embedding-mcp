#!/usr/bin/env bash
set -euo pipefail
context_run_dir=$(cd "$(dirname "$0")" && pwd)
context_build_dir="$PWD/.assets/mcp-build/source"
context_worker_dir=$(realpath .assets/embedding-worker/current)
g++ -std=c++17 -O2 -I"$context_build_dir/include" -I"$context_build_dir/ggml/include" -I"$context_build_dir/vendor" \
  "$context_run_dir/probe.cpp" -L"$context_worker_dir" -Wl,-rpath,"$context_worker_dir" -lllama -lggml -o "$context_run_dir/probe"
: > "$context_run_dir/summary.jsonl"
context_failed=0
for context_model in "$@"; do
  context_model_file=$(jq -r --arg model "$context_model" '.groups[$model][0].path' assets.lock.json)
  if "$context_run_dir/probe" "$context_model" "$PWD/.assets/$context_model_file" \
        > "$context_run_dir/$context_model.json" 2> "$context_run_dir/$context_model.log"; then
    cat "$context_run_dir/$context_model.json" >> "$context_run_dir/summary.jsonl"
  else
    tail -15 "$context_run_dir/$context_model.log" >&2
    context_failed=1
  fi
done
exit "$context_failed"
