@echo off
:: StreamScore 项目初始化脚本 (Windows)
echo === StreamScore Setup ===
echo.

:: 配置 Git Hooks
echo [1/2] 配置 Git Hooks...
git config core.hooksPath .githooks
echo  ✓ hooksPath = .githooks

:: 安装依赖
echo.
echo [2/2] 安装 Python 依赖...
pip install -r requirements.txt
echo  ✓ Dependencies installed

echo.
echo === Setup Complete ===
echo 每次 git commit 前会自动扫描密钥泄露。
