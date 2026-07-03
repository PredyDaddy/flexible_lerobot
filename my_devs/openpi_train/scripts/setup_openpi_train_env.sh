#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train"
OPENPI_ROOT="${ROOT}/openpi-main"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
OPENPI_CONDA_ENV="${OPENPI_CONDA_ENV:-openpi_train}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${OPENPI_CONDA_ENV}"; then
  conda create -y -n "${OPENPI_CONDA_ENV}" python=3.11
fi

conda activate "${OPENPI_CONDA_ENV}"
python -m pip install --upgrade pip

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install -U uv -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

cd "${OPENPI_ROOT}"
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e "packages/openpi-client"
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e "." --index-url https://pypi.tuna.tsinghua.edu.cn/simple

python - <<'PY'
import importlib.util
import sys

mods = ["jax", "flax", "orbax.checkpoint", "openpi", "openpi_client", "lerobot", "torch", "av"]
print("python", sys.version)
for mod in mods:
    spec = importlib.util.find_spec(mod)
    print(f"{mod}: {'OK' if spec else 'MISSING'} {spec.origin if spec else ''}")
PY

cat <<EOF

openpi_train environment is ready.
Use:
  source ${ROOT}/scripts/env.sh
EOF
