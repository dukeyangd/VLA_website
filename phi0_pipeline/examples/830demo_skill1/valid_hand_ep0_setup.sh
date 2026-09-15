#!/usr/bin/env bash
# 推理前置：valid-hand 软链 out ep0 → raw source ep1
set -euo pipefail

VH="${VALID_HAND_ROOT:-/tmp/830demo_skill1_valid_hand_ep0}"
SRC="${VALID_HAND_SRC:-/mnt/data2/wpy/workspace/830demo/skill_1_walk_to_black_box/2026-09-01-17-00-42/data/chunk-000/episode_000001.parquet}"

mkdir -p "${VH}/data/chunk-000"
ln -sfn "${SRC}" "${VH}/data/chunk-000/episode_000000.parquet"
echo "[valid_hand] ${VH}/data/chunk-000/episode_000000.parquet -> ${SRC}"
