#!/usr/bin/env bash
# Push Phi_0 markdown docs to Lark Wiki (my_library) mirroring local docs/ layout.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
# shellcheck disable=SC1091
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
command -v lark-cli >/dev/null || { echo "lark-cli not found; run: npx @larksuite/cli@latest install"; exit 1; }

WIKI_SPACE="${WIKI_SPACE:-my_library}"
WIKI_SPACE_ID="${WIKI_SPACE_ID:-}"
WIKI_ROOT_NAME="${WIKI_ROOT_NAME:-Phi-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
INSERT_IMAGES="${INSERT_IMAGES:-1}"
UPLOAD_DELAY_SEC="${UPLOAD_DELAY_SEC:-2}"

MAP_FILE="${ROOT}/.lark-docs-sync.map"
FOLDER_MAP_FILE="${ROOT}/.lark-docs-sync.folders"
IMG_MAP_FILE="${ROOT}/.lark-docs-sync.images"
STATE_DIR="${ROOT}/.lark-docs-sync"

usage() {
  cat <<EOF
Usage: sync_docs_to_lark.sh [command]

Commands:
  check       Verify lark-cli user login + scopes
  plan        List local md files, wiki target folders, image refs
  sync        Create docx under mirrored wiki folders + insert images (default)
  reorganize  Move already-synced docs into correct wiki folder tree
  folders     Create wiki folder tree only (mirrors docs/ layout)
  images      Insert images only (docs already in map)
  urls        Print synced doc URLs
  where       Where docs live in Lark + how to find them

See also: docs/lark_sync.md

Env:
  WIKI_ROOT_NAME=Phi-0       Wiki root folder name (default)
  WIKI_SPACE=my_library      Personal document library
  DRY_RUN=1                  Dry-run API calls
  SKIP_EXISTING=0            Re-create docx (usually use reorganize instead)
  INSERT_IMAGES=0            Skip image upload
  UPLOAD_DELAY_SEC=2         Rate-limit pause between API calls

Layout (mirrors repo):
  Phi-0/
    Phi-0 项目总览          <- README.md
    Pick-Tissue 部署指南    <- readme.md
    docs/
      report/
        architecture/
          action_head       <- docs/report/architecture/action_head.md
      ...
EOF
}

check_auth() {
  local st
  st="$(lark-cli auth status 2>&1)"
  echo "$st"
  if echo "$st" | grep -q '"user".*"available": false'; then
    echo "ERROR: user identity missing. Run:" >&2
    echo "  lark-cli auth login --recommend --domain docs,drive,wiki" >&2
    exit 1
  fi
}

print_where() {
  cat <<EOF
Phi_0 文档位置（按目录树同步）：

  类型：Lark 云文档（Docx），组织在 Wiki「我的文档库」
  根目录：${WIKI_ROOT_NAME}/
  租户：https://qjp6kv7ypf0e.jp.larksuite.com

在 Lark 里怎么找：
  1. 云文档 / Docs → 我的文档库（My Library）
  2. 打开文件夹「${WIKI_ROOT_NAME}」
  3. 目录结构与本地 docs/ 一致（docs/report/architecture/...）
  4. 或搜索文档标题；运行：bash tools/env/sync_docs_to_lark.sh urls

已有文档若还在库根平铺，运行：
  bash tools/env/sync_docs_to_lark.sh reorganize

映射：${MAP_FILE}
文件夹：${FOLDER_MAP_FILE}
EOF
}

collect_md() {
  {
    [ -f README.md ] && echo README.md
    [ -f readme.md ] && echo readme.md
    find docs -name '*.md' -type f | sort
  } | while read -r f; do
    case "$f" in docs/qwenNAV.md) continue ;; esac
    echo "$f"
  done
}

title_from_path() {
  local rel="$1"
  case "$rel" in
    README.md) echo "Phi-0 项目总览" ;;
    readme.md) echo "Phi-0 Pick-Tissue 部署指南" ;;
    docs/README.md) echo "Phi-0 文档索引" ;;
    docs/ecosystem/README.md) echo "Ecosystem 索引" ;;
    docs/lark_sync.md) echo "Lark 文档同步指南" ;;
    docs/config_reference.md) echo "配置索引" ;;
    docs/scripts_index.md) echo "常用脚本索引" ;;
    docs/testing.md) echo "测试建议" ;;
    docs/development/setup.md) echo "开发环境搭建" ;;
    docs/report/README.md) echo "Phi0 系统开发报告" ;;
    docs/report/eval.md) echo "Eval 与验收指南" ;;
    docs/report/benchmarks.md) echo "Benchmark 横向对比" ;;
    docs/report/training/data_pipeline.md) echo "数据管线" ;;
    docs/report/training/pick_yellow_box.md) echo "Pick-yellow-box 训练线" ;;
    docs/report/training/baseline_training_lines.md) echo "并行训练线与 Legacy" ;;
    docs/report/architecture/inference_session.md) echo "Inference Session 与 RTC" ;;
    docs/report/architecture/agent_module.md) echo "Agent 模块架构" ;;
    docs/report/deploy/zmq_protocol.md) echo "ZMQ 协议参考" ;;
    docs/report/deploy/robot_runbook.md) echo "真机 Sim 运行手册" ;;
    *) basename "${rel%.md}" ;;
  esac
}

# Wiki folder path for parent of this md (e.g. Phi-0/docs/report/architecture)
wiki_parent_path_for_rel() {
  local rel="$1"
  case "$rel" in
    README.md|readme.md) echo "${WIKI_ROOT_NAME}" ;;
    docs/*)
      local dir
      dir="$(dirname "$rel")"
      echo "${WIKI_ROOT_NAME}/${dir}"
      ;;
    *) echo "${WIKI_ROOT_NAME}" ;;
  esac
}

lookup_folder_token() {
  awk -F'\t' -v p="$1" '$1==p && NF>=2 && $2!=""{print $2; exit}' "$FOLDER_MAP_FILE" 2>/dev/null || true
}

record_folder_token() {
  local path="$1" token="$2"
  [ -n "$token" ] || return 1
  mkdir -p "$STATE_DIR"
  if [ -f "$FOLDER_MAP_FILE" ] && grep -q "^${path}	" "$FOLDER_MAP_FILE"; then
    awk -F'\t' -v p="$path" -v t="$token" '$1==p {print p "\t" t; next} {print}' \
      "$FOLDER_MAP_FILE" > "${FOLDER_MAP_FILE}.tmp" && mv "${FOLDER_MAP_FILE}.tmp" "$FOLDER_MAP_FILE"
  else
    printf '%s\t%s\n' "$path" "$token" >>"$FOLDER_MAP_FILE"
  fi
}

parse_lark_json_field() {
  local field="$1" text
  text="$(cat)"
  LARK_JSON_INPUT="$text" python3 - "$field" <<'PY'
import os, sys, json
field = sys.argv[1]
text = os.environ.get('LARK_JSON_INPUT', '')
dec = json.JSONDecoder()
idx = 0
while idx < len(text):
    while idx < len(text) and text[idx] not in '{[':
        idx += 1
    if idx >= len(text):
        break
    try:
        obj, end = dec.raw_decode(text, idx)
        idx = end
    except json.JSONDecodeError:
        idx += 1
        continue
    if isinstance(obj, dict) and obj.get('ok') and 'data' in obj:
        data = obj['data']
        if field == 'node_token':
            print(
                data.get('node_token')
                or data.get('wiki_token')
                or data.get('node', {}).get('node_token', '')
            )
        elif field == 'space_id':
            print(data.get('space_id') or data.get('space', {}).get('space_id', ''))
        elif field == 'doc_url':
            print(data.get('document', {}).get('url', ''))
        elif field == 'doc_id':
            print(data.get('document', {}).get('document_id', ''))
        break
PY
}

resolve_wiki_space_id() {
  [ -n "$WIKI_SPACE_ID" ] && return 0
  local out
  out="$(lark-cli wiki spaces get --as user --params "{\"space_id\":\"${WIKI_SPACE}\"}" 2>&1)" \
    || { echo "$out" >&2; return 1; }
  WIKI_SPACE_ID="$(echo "$out" | parse_lark_json_field space_id)"
  [ -n "$WIKI_SPACE_ID" ] || { echo "$out" >&2; return 1; }
}

find_child_node_by_title() {
  local parent_token="$1" title="$2"
  resolve_wiki_space_id || return 1
  local args=(wiki +node-list --as user --space-id "$WIKI_SPACE_ID" --page-all --page-limit 20)
  if [ -n "$parent_token" ]; then
    args+=(--parent-node-token "$parent_token")
  fi
  local out
  out="$(lark-cli "${args[@]}" 2>&1)" || return 1
  LARK_JSON_INPUT="$out" python3 - "$title" <<'PY'
import os, sys, json
title = sys.argv[1]
text = os.environ.get('LARK_JSON_INPUT', '')
dec = json.JSONDecoder()
idx = 0
while idx < len(text):
    while idx < len(text) and text[idx] not in '{[':
        idx += 1
    if idx >= len(text):
        break
    try:
        obj, end = dec.raw_decode(text, idx)
        idx = end
    except json.JSONDecodeError:
        idx += 1
        continue
    if isinstance(obj, dict) and obj.get('ok'):
        for item in obj.get('data', {}).get('items', []):
            if item.get('title') == title and item.get('node_token'):
                print(item['node_token'])
                raise SystemExit(0)
PY
}

create_wiki_folder_node() {
  local title="$1" parent_token="${2:-}" dry_args=()
  [ "$DRY_RUN" = "1" ] && dry_args=(--dry-run)
  local existing=""
  if [ -n "$parent_token" ]; then
    existing="$(find_child_node_by_title "$parent_token" "$title" 2>/dev/null || true)"
  fi
  if [ -n "$existing" ]; then
    echo "$existing"
    return 0
  fi
  local args=(wiki +node-create --as user --title "$title" "${dry_args[@]}")
  if [ -n "$parent_token" ]; then
    args+=(--parent-node-token "$parent_token")
  else
    args+=(--space-id "$WIKI_SPACE")
  fi
  local out token
  out="$(lark-cli "${args[@]}" 2>&1)" || { echo "$out" >&2; return 1; }
  token="$(echo "$out" | parse_lark_json_field node_token)"
  [ -n "$token" ] || { echo "$out" >&2; return 1; }
  echo "$token"
}

# Ensure Phi-0/docs/report/... folder chain exists; prints leaf folder node_token.
ensure_wiki_folder_path() {
  local full_path="$1"
  local existing
  existing="$(lookup_folder_token "$full_path")"
  [ -n "$existing" ] && { echo "$existing"; return 0; }

  local parent_token="" current="" part remaining="$full_path"
  while [ -n "$remaining" ]; do
    if [[ "$remaining" == */* ]]; then
      part="${remaining%%/*}"
      remaining="${remaining#*/}"
    else
      part="$remaining"
      remaining=""
    fi
    if [ -z "$current" ]; then
      current="$part"
    else
      current="${current}/${part}"
    fi
    existing="$(lookup_folder_token "$current")"
    if [ -n "$existing" ]; then
      parent_token="$existing"
      continue
    fi
    echo "FOLDER: $current" >&2
    parent_token="$(create_wiki_folder_node "$part" "$parent_token")"
    record_folder_token "$current" "$parent_token"
    api_sleep
  done
  echo "$parent_token"
}

lookup_map_url() { awk -F'\t' -v k="$1" '$1==k {print $2; exit}' "$MAP_FILE" 2>/dev/null || true; }
lookup_map_doc_token() { awk -F'\t' -v k="$1" '$1==k {print $3; exit}' "$MAP_FILE" 2>/dev/null || true; }
lookup_map_wiki_node() { awk -F'\t' -v k="$1" '$1==k {print $4; exit}' "$MAP_FILE" 2>/dev/null || true; }

record_map() {
  local key="$1" url="$2" doc_token="$3" wiki_node="${4:-}"
  mkdir -p "$STATE_DIR"
  if [ -f "$MAP_FILE" ] && grep -q "^${key}	" "$MAP_FILE"; then
    awk -F'\t' -v k="$key" -v u="$url" -v d="$doc_token" -v w="$wiki_node" \
      '$1==k {print k "\t" u "\t" d "\t" w; next} {print}' "$MAP_FILE" > "${MAP_FILE}.tmp" \
      && mv "${MAP_FILE}.tmp" "$MAP_FILE"
  else
    printf '%s\t%s\t%s\t%s\n' "$key" "$url" "$doc_token" "$wiki_node" >>"$MAP_FILE"
  fi
}

doc_in_wiki() {
  local doc_token="$1"
  local out
  out="$(lark-cli wiki +node-get --as user --node-token "$doc_token" --obj-type docx 2>&1)" || return 1
  echo "$out" | grep -q '"ok": true'
}

resolve_wiki_node_for_doc() {
  local doc_token="$1"
  local out
  out="$(lark-cli wiki +node-get --as user --node-token "$doc_token" --obj-type docx 2>&1)" || return 1
  echo "$out" | parse_lark_json_field node_token
}

image_already_inserted() {
  local md_rel="$1" img_rel="$2"
  [ -f "$IMG_MAP_FILE" ] || return 1
  awk -F'\t' -v m="$md_rel" -v i="$img_rel" '$1==m && $2==i && $3==1 {found=1; exit} END{exit !found}' "$IMG_MAP_FILE"
}

record_image_inserted() {
  local md_rel="$1" img_rel="$2"
  mkdir -p "$STATE_DIR"
  if [ -f "$IMG_MAP_FILE" ] && grep -q "^${md_rel}	${img_rel}	" "$IMG_MAP_FILE"; then
    awk -F'\t' -v m="$md_rel" -v i="$img_rel" '$1==m && $2==i {print m "\t" i "\t" 1; next} {print}' \
      "$IMG_MAP_FILE" > "${IMG_MAP_FILE}.tmp" && mv "${IMG_MAP_FILE}.tmp" "$IMG_MAP_FILE"
  else
    printf '%s\t%s\t1\n' "$md_rel" "$img_rel" >>"$IMG_MAP_FILE"
  fi
}

extract_image_refs() {
  local md_rel="$1"
  python3 - "$md_rel" "$ROOT" <<'PY'
import re, sys
from pathlib import Path

md_rel, root = sys.argv[1], Path(sys.argv[2])
md_path = root / md_rel
if not md_path.is_file():
    sys.exit(0)
text = md_path.read_text(encoding="utf-8", errors="replace")
md_dir = md_path.parent
current_heading = ""
seen = set()

def emit(anchor: str, src: str) -> None:
    if src.startswith(("http://", "https://", "data:")):
        return
    resolved = (md_dir / src.split("?")[0]).resolve()
    if not resolved.is_file():
        return
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return
    key = (anchor, str(rel))
    if key in seen:
        return
    seen.add(key)
    print(f"{anchor}\t{rel}")

for line in text.splitlines():
    hm = re.match(r"^#{1,6}\s+(.+)$", line)
    if hm:
        current_heading = hm.group(1).strip()
    for src in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', line, flags=re.I):
        alt_m = re.search(r'alt=["\']([^"\']*)["\']', line, flags=re.I)
        anchor = current_heading or (alt_m.group(1).strip() if alt_m else Path(src).stem)
        emit(anchor, src)
    for alt, src in re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', line):
        emit(current_heading or alt.strip() or Path(src).stem, src.strip())
PY
}

api_sleep() { sleep "$UPLOAD_DELAY_SEC"; }

insert_images_for_doc() {
  local md_rel="$1" doc_token="$2"
  [ "$INSERT_IMAGES" = "1" ] || return 0
  [ -n "$doc_token" ] || return 0
  local anchor img_rel dry_args=()
  [ "$DRY_RUN" = "1" ] && dry_args=(--dry-run)
  while IFS=$'\t' read -r anchor img_rel; do
    [ -n "$anchor" ] && [ -n "$img_rel" ] || continue
    [ -f "$img_rel" ] || { echo "  WARN: missing $img_rel" >&2; continue; }
    if image_already_inserted "$md_rel" "$img_rel"; then
      echo "  SKIP image: $img_rel"
      continue
    fi
    echo "  IMAGE: $img_rel  anchor=$anchor"
    if lark-cli docs +media-insert --as user --doc "$doc_token" --file "$img_rel" \
        --type image --align center --selection-with-ellipsis "$anchor" "${dry_args[@]}" \
        >/dev/null 2>&1; then
      [ "$DRY_RUN" != "1" ] && record_image_inserted "$md_rel" "$img_rel"
      echo "    ok"
    elif lark-cli docs +media-insert --as user --doc "$doc_token" --file "$img_rel" \
        --type image --align center "${dry_args[@]}" >/dev/null 2>&1; then
      [ "$DRY_RUN" != "1" ] && record_image_inserted "$md_rel" "$img_rel"
      echo "    ok (appended)"
    else
      echo "    FAILED: $img_rel" >&2
    fi
    api_sleep
  done < <(extract_image_refs "$md_rel")
}

upload_one() {
  local rel="$1"
  local parent_path parent_token title dry_args=()
  local existing_url existing_doc

  existing_url="$(lookup_map_url "$rel")"
  existing_doc="$(lookup_map_doc_token "$rel")"
  parent_path="$(wiki_parent_path_for_rel "$rel")"
  parent_token="$(ensure_wiki_folder_path "$parent_path")"

  if [ "$SKIP_EXISTING" = "1" ] && [ -n "$existing_url" ]; then
    echo "SKIP create: $rel -> $existing_url  (wiki: $parent_path/)"
    insert_images_for_doc "$rel" "$existing_doc"
    return 0
  fi

  title="$(title_from_path "$rel")"
  [ "$DRY_RUN" = "1" ] && dry_args=(--dry-run)

  echo "UPLOAD: $rel  title=$title  folder=$parent_path/"
  local out url doc_token wiki_node
  out="$(lark-cli docs +create --as user --doc-format markdown --title "$title" \
    --content "@${rel}" --parent-token "$parent_token" "${dry_args[@]}" 2>&1)" || {
    echo "$out" >&2; echo "FAILED: $rel" >&2; return 1
  }
  if [ "$DRY_RUN" = "1" ]; then echo "$out"; return 0; fi

  url="$(echo "$out" | parse_lark_json_field doc_url)"
  doc_token="$(echo "$out" | parse_lark_json_field doc_id)"
  [ -n "$url" ] && [ -n "$doc_token" ] || { echo "$out"; return 0; }
  wiki_node="$(resolve_wiki_node_for_doc "$doc_token" 2>/dev/null || true)"
  record_map "$rel" "$url" "$doc_token" "$wiki_node"
  echo "  -> $url"
  api_sleep
  insert_images_for_doc "$rel" "$doc_token"
}

reorganize_one() {
  local rel="$1"
  local doc_token wiki_node parent_path target_parent dry_args=() out moved=0
  doc_token="$(lookup_map_doc_token "$rel")"
  [ -n "$doc_token" ] || { echo "SKIP (not synced): $rel"; return 0; }

  resolve_wiki_space_id || return 1
  parent_path="$(wiki_parent_path_for_rel "$rel")"
  target_parent="$(ensure_wiki_folder_path "$parent_path")"
  [ -n "$target_parent" ] || { echo "WARN: no target folder for $rel ($parent_path)" >&2; return 1; }

  wiki_node="$(lookup_map_wiki_node "$rel")"
  [ -n "$wiki_node" ] || wiki_node="$(resolve_wiki_node_for_doc "$doc_token" 2>/dev/null || true)"

  [ "$DRY_RUN" = "1" ] && dry_args=(--dry-run)
  echo "MOVE: $rel -> $parent_path/  doc=$doc_token wiki=${wiki_node:-new}"

  if [ -n "$wiki_node" ]; then
    if lark-cli wiki +move --as user --node-token "$wiki_node" \
        --target-parent-token "$target_parent" "${dry_args[@]}" >/dev/null 2>&1; then
      moved=1
    fi
  elif doc_in_wiki "$doc_token" 2>/dev/null; then
    wiki_node="$(resolve_wiki_node_for_doc "$doc_token")"
    if lark-cli wiki +move --as user --node-token "$wiki_node" \
        --target-parent-token "$target_parent" "${dry_args[@]}" >/dev/null 2>&1; then
      moved=1
    fi
  else
    out="$(lark-cli wiki +move --as user --obj-type docx --obj-token "$doc_token" \
      --target-space-id "$WIKI_SPACE_ID" --target-parent-token "$target_parent" \
      "${dry_args[@]}" 2>&1)" || { echo "$out" >&2; echo "  FAILED: $rel" >&2; api_sleep; return 1; }
    wiki_node="$(echo "$out" | parse_lark_json_field node_token)"
    moved=1
  fi

  if [ "$moved" = "1" ]; then
    [ -n "$wiki_node" ] || wiki_node="$(resolve_wiki_node_for_doc "$doc_token" 2>/dev/null || true)"
    record_map "$rel" "$(lookup_map_url "$rel")" "$doc_token" "$wiki_node"
    echo "  ok"
  else
    echo "  FAILED or already in place: $rel" >&2
  fi
  api_sleep
}

folders_only() {
  check_auth
  mkdir -p "$STATE_DIR"
  echo "Building wiki folder tree..."
  build_folder_tree
  echo "Folder map: $FOLDER_MAP_FILE"
  column -t -s $'\t' "$FOLDER_MAP_FILE" 2>/dev/null || cat "$FOLDER_MAP_FILE"
}

reorganize_all() {
  check_auth
  mkdir -p "$STATE_DIR"
  echo "Building wiki folder tree..."
  build_folder_tree
  local fail=0 n=0
  while IFS= read -r f; do
    n=$((n + 1))
    reorganize_one "$f" || fail=$((fail + 1))
  done < <(collect_md)
  echo "Done reorganize: $n docs, $fail failed."
}

build_folder_tree() {
  mkdir -p "$STATE_DIR"
  resolve_wiki_space_id || return 1
  local paths p
  paths=""
  while IFS= read -r f; do
    paths+="$(wiki_parent_path_for_rel "$f")"$'\n'
  done < <(collect_md)
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    ensure_wiki_folder_path "$p" >/dev/null
  done < <(printf '%s' "$paths" | sort -u)
}

plan() {
  echo "Wiki layout under ${WIKI_ROOT_NAME}/ (from $ROOT):"
  while IFS= read -r f; do
    echo "  $f"
    echo "    -> $(wiki_parent_path_for_rel "$f")/"
    while IFS=$'\t' read -r anchor img; do
      echo "    img: $img (anchor: $anchor)"
    done < <(extract_image_refs "$f")
  done < <(collect_md)
}

sync_all() {
  check_auth
  mkdir -p "$STATE_DIR"
  echo "Building wiki folder tree..."
  build_folder_tree
  local n=0 fail=0
  while IFS= read -r f; do
    n=$((n + 1))
    upload_one "$f" || fail=$((fail + 1))
  done < <(collect_md)
  echo "Done: $n files, $fail failed. Map: $MAP_FILE"
}

images_only() {
  check_auth
  [ -f "$MAP_FILE" ] || exit 1
  while IFS=$'\t' read -r rel _url doc _wiki; do
    [ -n "$rel" ] || continue
    echo "IMAGES: $rel"
    insert_images_for_doc "$rel" "$doc"
  done < "$MAP_FILE"
}

print_urls() {
  [ -f "$MAP_FILE" ] || exit 1
  echo -e "path\turl\tfolder"
  while IFS=$'\t' read -r rel url _doc _wiki; do
    [ -n "$rel" ] || continue
    printf '%s\t%s\t%s/\n' "$rel" "$url" "$(wiki_parent_path_for_rel "$rel")"
  done < "$MAP_FILE" | column -t -s $'\t'
}

cmd="${1:-sync}"
case "$cmd" in
  check) check_auth ;;
  plan) plan ;;
  sync) sync_all ;;
  reorganize) reorganize_all ;;
  folders) folders_only ;;
  images) images_only ;;
  urls) print_urls ;;
  where) print_where ;;
  -h|--help|help) usage ;;
  *) echo "Unknown: $cmd"; usage; exit 1 ;;
esac
