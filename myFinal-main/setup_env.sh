#!/usr/bin/env bash
# Create conda env for L2 speaker-verification experiments.
set -euo pipefail

ENV_NAME="${ENV_NAME:-l2-spkverif}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${1:-auto}"  # auto | cuda | cpu

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Load your conda module first, e.g.:"
  echo "  module load anaconda3"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists; reusing it."
else
  echo "Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION}) ..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

if [[ "${DEVICE}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

echo "Installing PyTorch for: ${DEVICE}"
if [[ "${DEVICE}" == "cuda" ]]; then
  pip install --upgrade pip
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
else
  pip install --upgrade pip
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing project requirements ..."
pip install -r "${PROJECT_ROOT}/requirements.txt"

mkdir -p "${PROJECT_ROOT}/hf_cache"

echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Optional (keeps HF cache on scratch):"
echo "  export HF_CACHE_ROOT=${PROJECT_ROOT}/hf_cache"
echo ""
echo "Next steps:"
echo "  hf auth login"
echo "  python download.py"
echo "  python run_eval.py --device ${DEVICE}"
