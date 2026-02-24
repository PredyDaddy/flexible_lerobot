#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-${WORKSPACE_ROOT}/lerobot-0.4.3}"

if [[ ! -f "${LEROBOT_DIR}/pyproject.toml" ]]; then
  echo "[error] 未找到 lerobot 源码目录: ${LEROBOT_DIR}" >&2
  echo "请设置 LEROBOT_DIR 指向包含 pyproject.toml 的目录。" >&2
  exit 1
fi

echo "[info] 使用本地源码目录: ${LEROBOT_DIR}"
echo "[info] 卸载可能存在的 PyPI 版本 lerobot"
python -m pip uninstall -y lerobot >/dev/null 2>&1 || true

echo "[info] 安装 editable 版本 (pip install -e)"
python -m pip install -e "${LEROBOT_DIR}"

echo "[info] 校验当前导入路径"
python - <<'PY'
import pathlib
import lerobot

print(f"[ok] lerobot import path: {pathlib.Path(lerobot.__file__).resolve()}")
PY

