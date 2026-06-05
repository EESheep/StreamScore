#!/usr/bin/env bash
# StreamScore 项目初始化脚本 (Linux/macOS)
set -euo pipefail

echo "=== StreamScore Setup ==="
echo ""

# 配置 Git Hooks
echo "[1/2] Configuring Git Hooks..."
git config core.hooksPath .githooks
echo "  ✓ hooksPath = .githooks"

# 安装依赖
echo ""
echo "[2/2] Installing Python dependencies..."
pip install -r requirements.txt
echo "  ✓ Dependencies installed"

echo ""
echo "=== Setup Complete ==="
echo "每次 git commit 前会自动扫描密钥泄露。"
