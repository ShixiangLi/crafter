#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_ROOT"

# Bypass proxies for local services in this process tree only.
# Merge both spellings so existing bypass rules remain effective.
PROJECT_NO_PROXY="127.0.0.1,localhost,::1"
if [[ -n "${NO_PROXY:-}" ]]; then PROJECT_NO_PROXY+=",$NO_PROXY"; fi
if [[ -n "${no_proxy:-}" ]]; then PROJECT_NO_PROXY+=",$no_proxy"; fi
export NO_PROXY="$PROJECT_NO_PROXY"
export no_proxy="$PROJECT_NO_PROXY"

GPU=""
ARGS=()
DRY_RUN=false
SHOW_HELP=false
while (($#)); do
  case "$1" in
    --gpu|--gpu=*)
      [[ -z "$GPU" ]] || { echo "Specify --gpu only once." >&2; exit 2; }
      if [[ "$1" == --gpu=* ]]; then
        GPU="${1#*=}"
      else
        (($# >= 2)) || { echo "--gpu requires a GPU index." >&2; exit 2; }
        GPU="$2"
        shift
      fi
      [[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu requires a nonnegative GPU index." >&2; exit 2; }
      ;;
    --dry-run) DRY_RUN=true; ARGS+=("$1") ;;
    -h|--help) SHOW_HELP=true; ARGS+=("$1") ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done

if "$SHOW_HELP"; then
  echo "Launcher option: --gpu N starts/reuses a GPU-specific Ollama on port 11500+N."
  echo "Without --gpu, use the configured API service. --dry-run never starts a service."
elif [[ -n "$GPU" ]]; then
  for arg in "${ARGS[@]}"; do
    [[ "$arg" != *client.base_url=* ]] || { echo "Do not combine --gpu with client.base_url." >&2; exit 2; }
  done
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required for --gpu." >&2; exit 2; }
  GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F ', *' -v gpu="$GPU" '$1 == gpu {print $2}')"
  [[ -n "$GPU_UUID" ]] || { echo "GPU $GPU not found; check nvidia-smi -L." >&2; exit 2; }
  GPU=$((10#$GPU))
  PORT=$((11500 + GPU))
  HOST="127.0.0.1:$PORT"
  ARGS+=("--ollama-gpu" "client.base_url=http://$HOST/v1")
  if ! "$DRY_RUN"; then
    # Resolve the model from the effective experiment config before service startup.
    MODEL_DIR="$("$PROJECT_ROOT/.venv/bin/python" -B "$PROJECT_ROOT/scripts/ollama_models.py" --directory "${ARGS[@]}")"
    for tool in ollama curl flock; do
      command -v "$tool" >/dev/null || { echo "$tool is required for --gpu." >&2; exit 2; }
    done
    SERVICE_DIR="$PROJECT_ROOT/outputs/ollama"
    mkdir -p "$SERVICE_DIR"
    PID_FILE="$SERVICE_DIR/gpu_$GPU.pid"
    LOG_FILE="$SERVICE_DIR/gpu_$GPU.log"
    (
      flock -x 9
      PID="$(cat "$PID_FILE" 2>/dev/null || true)"
      if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null &&
         grep -zFxq "CUDA_VISIBLE_DEVICES=$GPU_UUID" "/proc/$PID/environ" 2>/dev/null &&
         grep -zFxq "OLLAMA_HOST=$HOST" "/proc/$PID/environ" 2>/dev/null &&
         grep -zFxq "serve" "/proc/$PID/cmdline" 2>/dev/null; then
        curl --noproxy '*' -fsS --max-time 3 "http://$HOST/api/version" >/dev/null || {
          echo "Managed Ollama is not ready; inspect $LOG_FILE (PID $PID)." >&2; exit 1;
        }
        # Do not silently reuse a server with a different model store.
        if ! grep -zFxq "OLLAMA_MODELS=$MODEL_DIR" "/proc/$PID/environ" &&
           ! { ! grep -zq '^OLLAMA_MODELS=' "/proc/$PID/environ" &&
               [[ "$MODEL_DIR" == "$HOME/.ollama/models" ]] &&
               grep -zFxq "HOME=$HOME" "/proc/$PID/environ"; }; then
          echo "Managed Ollama uses another model directory. Stop PID $PID when idle, then rerun with OLLAMA_MODELS=$MODEL_DIR." >&2
          exit 1
        fi
        echo "Reusing Ollama: GPU $GPU, $HOST, PID $PID; models: $MODEL_DIR" >&2
      else
        if (echo >"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
          echo "Port $PORT is occupied by an unverified service; refusing to reuse it." >&2
          exit 1
        fi
        # GPU UUIDs avoid CUDA/NVML numeric ordering differences. Disable Vulkan fallback.
        CUDA_VISIBLE_DEVICES="$GPU_UUID" OLLAMA_HOST="$HOST" OLLAMA_VULKAN=0 \
          OLLAMA_MODELS="$MODEL_DIR" OLLAMA_NOPRUNE=true \
          nohup ollama serve >>"$LOG_FILE" 2>&1 < /dev/null 9>&- &
        PID=$!
        echo "$PID" >"$PID_FILE"
        READY=false
        for ((attempt=0; attempt<60; attempt++)); do
          kill -0 "$PID" 2>/dev/null || break
          if curl --noproxy '*' -fsS --max-time 1 "http://$HOST/api/version" >/dev/null 2>&1; then
            READY=true
            break
          fi
          sleep 1
        done
        if ! "$READY"; then
          kill "$PID" 2>/dev/null || true
          wait "$PID" 2>/dev/null || true
          rm -f "$PID_FILE"
          echo "Ollama startup failed; inspect $LOG_FILE." >&2
          exit 1
        fi
        echo "Started Ollama: GPU $GPU, $HOST, PID $PID; log: $LOG_FILE" >&2
      fi
    ) 9>"$SERVICE_DIR/gpu_$GPU.lock"
    "$PROJECT_ROOT/.venv/bin/python" -B "$PROJECT_ROOT/scripts/ollama_models.py" --check "${ARGS[@]}"
  fi
fi

exec "$PROJECT_ROOT/.venv/bin/python" -B "$PROJECT_ROOT/scripts/evaluate.py" "${ARGS[@]}"
