#!/usr/bin/env bash
# 打可跨机部署的源码包：不含 .venv / 权重 / state / experiments / 本机软链。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d)}"
NAME="wb-vla-data-station-src-${STAMP}"
OUT_DIR="${OUT_DIR:-$ROOT}"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/${NAME}.XXXXXX")"
DEST="$STAGE/$NAME"

cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

mkdir -p "$DEST"

# 根目录可迁移文件
for f in \
  app.py collect_backend.py infer_backend.py skill_backend.py train_backend.py \
  start.sh requirements.txt .gitignore \
  README.md template.html
do
  if [[ -e "$ROOT/$f" ]]; then
    cp -a "$ROOT/$f" "$DEST/"
  fi
done

# 目录：rsync 排除运行时与本机软链
rsync -a \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.venv/' \
  --exclude 'experiments/' \
  --exclude 'src' \
  --exclude 'subpackages' \
  --exclude '.venv_sim' \
  "$ROOT/static" "$ROOT/tests" "$ROOT/docs" "$ROOT/sonic_qa" \
  "$ROOT/data" "$ROOT/scripts" \
  "$DEST/"

mkdir -p "$DEST/phi0_pipeline"
rsync -a \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'experiments/' \
  --exclude 'src' \
  --exclude 'subpackages' \
  --exclude '.venv_sim' \
  "$ROOT/phi0_pipeline/" "$DEST/phi0_pipeline/"

# 空占位，避免误带大缓存
mkdir -p "$DEST/cache" "$DEST/phi0_pipeline/experiments"
if [[ -f "$ROOT/cache/README.md" ]]; then
  cp -a "$ROOT/cache/README.md" "$DEST/cache/"
else
  printf '# 缓存目录（可空）\n' > "$DEST/cache/README.md"
fi
if [[ -f "$ROOT/phi0_pipeline/LINKS.md" ]]; then
  cp -a "$ROOT/phi0_pipeline/LINKS.md" "$DEST/phi0_pipeline/"
fi
touch "$DEST/phi0_pipeline/experiments/.gitkeep"

# 确保 README 在包内
[[ -f "$DEST/README.md" ]] || { echo "缺少 README.md" >&2; exit 1; }

TAR="$OUT_DIR/${NAME}.tar.gz"
ZIP="$OUT_DIR/${NAME}.zip"
rm -f "$TAR" "$ZIP"
tar -C "$STAGE" -czf "$TAR" "$NAME"
( cd "$STAGE" && zip -qr "$ZIP" "$NAME" )

echo "OK: $TAR"
echo "OK: $ZIP"
du -h "$TAR" "$ZIP"
find "$DEST" -type f | wc -l | awk '{print "files:", $1}'
