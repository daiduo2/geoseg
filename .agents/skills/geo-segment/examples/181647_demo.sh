#!/usr/bin/env bash
set -e
# 181647 三联图烟雾测试
# 输入: photo/微信图片_20260513181647_74_251.jpg
# VLM:  复用 Phase 0 的 in-session Kimi 结果
# 输出: out/微信图片_20260513181647_74_251/ 目录

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMG="${SCRIPT_DIR}/../../photo/微信图片_20260513181647_74_251.jpg"
VLM="${SCRIPT_DIR}/../../photo/phase0/kimi_in_session.json"

echo "=== geo-segment smoke test (181647 top panel) ==="
echo "Image: ${IMG}"
echo "VLM JSON: ${VLM}"

if [[ ! -f "$IMG" ]]; then
    echo "Missing image: $IMG"
    exit 1
fi
if [[ ! -f "$VLM" ]]; then
    echo "Missing VLM JSON: $VLM"
    exit 1
fi

cd "$SCRIPT_DIR"
python3 -m lib.cli "$IMG" --vlm-json "$VLM" --panel=0 --zones=5 --interactive

echo ""
echo "=== Expected outputs ==="
ls -1 out/微信图片_20260513181647_74_251/ 2>/dev/null || true
