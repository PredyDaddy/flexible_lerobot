#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/test/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

exec python -m cqy.allign.head_alignment_service "$@"
