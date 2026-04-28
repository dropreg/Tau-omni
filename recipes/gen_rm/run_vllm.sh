#!/usr/bin/env bash

set -euo pipefail

ACTION=${1:-start}

# Local Qwen model path.
MODEL_PATH=${MODEL_PATH:-/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen3-8B}

# Port layout.
START_PORT=${START_PORT:-8000}
NUM_INSTANCES=${NUM_INSTANCES:-8}

# GPU layout.
GPUS_PER_INSTANCE=${GPUS_PER_INSTANCE:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}

# vLLM runtime options.
HOST=${HOST:-0.0.0.0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-true}
LOG_DIR=${LOG_DIR:-recipes/gen_rm/logs}
PID_DIR=${PID_DIR:-recipes/gen_rm/pids}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$LOG_DIR" "$PID_DIR"

extra_args_array=()
if [[ -n "$EXTRA_ARGS" ]]; then
  # EXTRA_ARGS uses normal shell word splitting, for example:
  # EXTRA_ARGS="--served-model-name qwen-genrm --max-num-seqs 64"
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
fi

if [[ -z "$GPU_IDS" ]]; then
  echo "GPU_IDS is empty. Set GPU_IDS or CUDA_VISIBLE_DEVICES first."
  echo "Example: GPU_IDS=0,1,2,3 NUM_INSTANCES=2 GPUS_PER_INSTANCE=2 bash recipes/gen_rm/run_vllm.sh start"
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
TOTAL_GPUS=${#GPU_ARRAY[@]}
REQUIRED_GPUS=$((NUM_INSTANCES * GPUS_PER_INSTANCE))

require_gpu_layout() {
  if (( TOTAL_GPUS < REQUIRED_GPUS )); then
    echo "Not enough GPUs."
    echo "  visible GPUs: $GPU_IDS"
    echo "  required GPUs: $REQUIRED_GPUS"
    echo "  num instances: $NUM_INSTANCES"
    echo "  gpus per instance: $GPUS_PER_INSTANCE"
    exit 1
  fi
}

pid_file_for_port() {
  local port=$1
  echo "$PID_DIR/vllm_port_${port}.pid"
}

log_file_for_port() {
  local port=$1
  echo "$LOG_DIR/vllm_port_${port}.log"
}

is_pid_running() {
  local pid=$1
  kill -0 "$pid" >/dev/null 2>&1
}

is_port_listening() {
  local port=$1
  lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1
}

stop_one_port() {
  local port=$1
  local pid_file
  pid_file=$(pid_file_for_port "$port")
  local stopped=false

  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(cat "$pid_file")
    if [[ -n "$pid" ]] && is_pid_running "$pid"; then
      echo "Stopping port $port, pid=$pid"
      kill "$pid" >/dev/null 2>&1 || true
      for _ in {1..20}; do
        if ! is_pid_running "$pid"; then
          stopped=true
          break
        fi
        sleep 1
      done
      if is_pid_running "$pid"; then
        echo "Force killing port $port, pid=$pid"
        kill -9 "$pid" >/dev/null 2>&1 || true
      fi
      stopped=true
    fi
    rm -f "$pid_file"
  fi

  if is_port_listening "$port"; then
    local port_pids
    port_pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN -n -P || true)
    if [[ -n "$port_pids" ]]; then
      echo "Stopping leftover process on port $port: $port_pids"
      # shellcheck disable=SC2086
      kill $port_pids >/dev/null 2>&1 || true
      sleep 2
      port_pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN -n -P || true)
      if [[ -n "$port_pids" ]]; then
        echo "Force killing leftover process on port $port: $port_pids"
        # shellcheck disable=SC2086
        kill -9 $port_pids >/dev/null 2>&1 || true
      fi
      stopped=true
    fi
  fi

  if [[ "$stopped" == false ]]; then
    echo "Port $port is not running"
  fi
}

start_all() {
  require_gpu_layout

  echo "Model path: $MODEL_PATH"
  echo "Start port: $START_PORT"
  echo "Instances : $NUM_INSTANCES"
  echo "GPU layout: $GPU_IDS"
  echo "GPUs/inst : $GPUS_PER_INSTANCE"
  echo "Log dir   : $LOG_DIR"
  echo "PID dir   : $PID_DIR"
  echo

  for ((i = 0; i < NUM_INSTANCES; i++)); do
    local port=$((START_PORT + i))
    local begin=$((i * GPUS_PER_INSTANCE))
    local instance_gpus=("${GPU_ARRAY[@]:begin:GPUS_PER_INSTANCE}")
    local cuda_devices
    cuda_devices=$(IFS=,; echo "${instance_gpus[*]}")
    local log_file
    log_file=$(log_file_for_port "$port")
    local pid_file
    pid_file=$(pid_file_for_port "$port")

    if [[ -f "$pid_file" ]]; then
      local old_pid
      old_pid=$(cat "$pid_file")
      if [[ -n "$old_pid" ]] && is_pid_running "$old_pid"; then
        echo "Port $port already running, pid=$old_pid"
        continue
      fi
      rm -f "$pid_file"
    fi

    if is_port_listening "$port"; then
      echo "Port $port is already occupied. Stop it first or change START_PORT."
      exit 1
    fi

    echo "Launching instance $i on port $port with GPUs [$cuda_devices]"

    CUDA_VISIBLE_DEVICES="$cuda_devices" \
    python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --host "$HOST" \
      --port "$port" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      $([[ "$TRUST_REMOTE_CODE" == "true" ]] && echo "--trust-remote-code") \
      "${extra_args_array[@]}" \
      > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    echo "  log: $log_file"
    echo "  pid: $pid"
  done

  echo
  echo "All requested vLLM instances have been started."
}

stop_all() {
  for ((i = 0; i < NUM_INSTANCES; i++)); do
    stop_one_port $((START_PORT + i))
  done
}

status_all() {
  for ((i = 0; i < NUM_INSTANCES; i++)); do
    local port=$((START_PORT + i))
    local pid_file
    pid_file=$(pid_file_for_port "$port")

    if [[ -f "$pid_file" ]]; then
      local pid
      pid=$(cat "$pid_file")
      if [[ -n "$pid" ]] && is_pid_running "$pid"; then
        echo "RUNNING  port=$port pid=$pid"
        continue
      fi
      echo "STALE    port=$port pid_file=$pid_file"
      continue
    fi

    if is_port_listening "$port"; then
      echo "RUNNING  port=$port pid=unknown"
    else
      echo "STOPPED  port=$port"
    fi
  done
}

restart_all() {
  stop_all
  start_all
}

case "$ACTION" in
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  restart)
    restart_all
    ;;
  status)
    status_all
    ;;
  *)
    echo "Usage: bash recipes/gen_rm/run_vllm.sh {start|stop|restart|status}"
    exit 1
    ;;
esac
