#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load local overrides first so the defaults below never overwrite user choices.
# Keep .env shell-compatible (KEY=value). The file is gitignored.
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

# -----------------------------------------------------------------------------
# Mac mini M4 / 16GB defaults
# -----------------------------------------------------------------------------
# Keep the 0.6B LM on MLX, but avoid holding a second MLX DiT copy alongside
# the PyTorch/MPS DiT. For long audio, keep VAE decode on tiled PyTorch/MPS.
export ACESTEP_LM_BACKEND="${ACESTEP_LM_BACKEND:-mlx}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-0.6B}"
export ACESTEP_INIT_LLM="${ACESTEP_INIT_LLM:-auto}"
export ACESTEP_OFFLOAD_TO_CPU="${ACESTEP_OFFLOAD_TO_CPU:-false}"
export ACESTEP_OFFLOAD_DIT_TO_CPU="${ACESTEP_OFFLOAD_DIT_TO_CPU:-false}"
export ACESTEP_LM_OFFLOAD_TO_CPU="${ACESTEP_LM_OFFLOAD_TO_CPU:-false}"
export ACESTEP_USE_MLX_DIT="${ACESTEP_USE_MLX_DIT:-false}"
export ACESTEP_MLX_VAE="${ACESTEP_MLX_VAE:-0}"
export ACESTEP_MLX_VAE_CHUNK="${ACESTEP_MLX_VAE_CHUNK:-192}"
export ACESTEP_COMPILE_MODEL="${ACESTEP_COMPILE_MODEL:-false}"
export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"

# The API server is an in-memory queue/store and is designed for one worker.
# Serial generation is also much safer on 16GB unified memory than parallel jobs.
export ACESTEP_QUEUE_WORKERS="${ACESTEP_QUEUE_WORKERS:-1}"
export ACESTEP_QUEUE_MAXSIZE="${ACESTEP_QUEUE_MAXSIZE:-32}"
export ACESTEP_NO_INIT="${ACESTEP_NO_INIT:-false}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Portable cache defaults. Override ACESTEP_CACHE_ROOT in .env to put all
# caches on a fast external NVMe/Thunderbolt SSD, e.g. /Volumes/ssd/ace-step.
CACHE_ROOT="${ACESTEP_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/Library/Caches}/ace-step}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$CACHE_ROOT/uv-python}"
mkdir -p "$MODELSCOPE_CACHE" "$HUGGINGFACE_HUB_CACHE" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

export ACESTEP_DOWNLOAD_SOURCE="${ACESTEP_DOWNLOAD_SOURCE:-modelscope}"
export ACESTEP_API_HOST="${ACESTEP_API_HOST:-127.0.0.1}"
export ACESTEP_API_PORT="${ACESTEP_API_PORT:-8001}"

# Prefer a normal uv install location without baking one machine's absolute path
# into the repository.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "[macOS] Warning: this launcher is tuned for Apple Silicon (arm64)."
fi

PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "[macOS] Missing $PYTHON"
  echo "[macOS] Run ./start_gradio_ui_macos.sh once, or create the environment with uv sync."
  exit 1
fi

# Lightweight preflight: warn about common unified-memory competitors.
if command -v pgrep >/dev/null 2>&1; then
  heavy=""
  for proc in ollama Docker qemu-system-aarch64 Unity Android\ Studio; do
    if pgrep -f "$proc" >/dev/null 2>&1; then
      heavy="$heavy $proc"
    fi
  done
  if [[ -n "$heavy" ]]; then
    echo "[macOS] Warning: memory-heavy processes detected:$heavy"
    echo "[macOS] For long 60-90s generations, close them to reduce swap pressure."
  fi
fi

echo "[macOS] ACE-Step performance profile"
echo "  LM backend/model : $ACESTEP_LM_BACKEND / $ACESTEP_LM_MODEL_PATH"
echo "  MLX DiT          : $ACESTEP_USE_MLX_DIT"
echo "  MLX VAE          : $ACESTEP_MLX_VAE (chunk=$ACESTEP_MLX_VAE_CHUNK)"
echo "  CPU offload      : $ACESTEP_OFFLOAD_TO_CPU"
echo "  Queue workers    : $ACESTEP_QUEUE_WORKERS"
echo "  Cache root       : $CACHE_ROOT"
echo "  API              : http://$ACESTEP_API_HOST:$ACESTEP_API_PORT"
echo

exec "$PYTHON" -m acestep.api_server \
  --host "$ACESTEP_API_HOST" \
  --port "$ACESTEP_API_PORT" \
  --lm-model-path "$ACESTEP_LM_MODEL_PATH" \
  --download-source "$ACESTEP_DOWNLOAD_SOURCE"
