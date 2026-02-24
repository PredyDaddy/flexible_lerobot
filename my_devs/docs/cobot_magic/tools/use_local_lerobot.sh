#!/usr/bin/env bash
# 需要用 `source` 执行，才会把 PYTHONPATH 注入当前 shell。

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "请使用: source cobot_magic/tools/use_local_lerobot.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-${WORKSPACE_ROOT}/lerobot-0.4.3}"
LEROBOT_SRC="${LEROBOT_DIR}/src"

if [[ ! -d "${LEROBOT_SRC}" ]]; then
  echo "[error] 未找到 lerobot 源码目录: ${LEROBOT_SRC}" >&2
  echo "请先确认目录存在，或设置 LEROBOT_DIR 后重试。" >&2
  return 1
fi

if [[ ":${PYTHONPATH:-}:" != *":${LEROBOT_SRC}:"* ]]; then
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH}"
  else
    export PYTHONPATH="${LEROBOT_SRC}"
  fi
fi

echo "[ok] 已启用本地 lerobot 源码优先:"
echo "     ${LEROBOT_SRC}"
echo "[hint] 当前 shell 已生效。可执行:"
echo "       python -c 'import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).resolve())'"
