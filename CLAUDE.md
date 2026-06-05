# StreamScore 开发指南

## 架构文档
- 领域语言 + 决策记录: `CONTEXT.md`
- 模块结构 + 数据流: `ARCHITECTURE.md`
- 全局配置: `config.yaml`

## 开发原则
- 每次修改只触及一个模块文件，避免跨模块改动
- 同级模块之间零 import，通过 `__init__.py` 编排
- Prompt 在 `prompts/` 目录独立管理，不与 Python 代码混合
- GPU 模型在模块内部自行管理加载/卸载，调用方无需感知
- 所有中间文件格式统一为 JSONL

## 运行
- 声纹注册（前置）: `python -m voiceprint_server.server --port 8080`
- 主控流水线: `python run_pipeline.py --room_id <id> --date <date>`
- 调试模式: `python run_pipeline.py --room_id <id> --date <date> --skip-enhance --skip-clip`

## 安全
- **强制**：pre-commit hook 自动扫描密钥泄露，阻止提交含 API Key/Token 的代码
- 新开发者克隆后必须运行 `bash scripts/setup.sh`（或 Windows: `scripts\setup.bat`）初始化 hooks
- 配置文件中的 API Key 使用占位符（如 `sk-your-api-key`），真实密钥通过环境变量注入
- 不得绕过 hook（`--no-verify`），除非经代码审查确认安全

## 测试
- 单模块调试: `python -m modules.preprocess --flv <path>`
- 目标硬件: NVIDIA RTX 3060 Laptop (8GB 显存)
- Python 3.10+, CUDA 11.8+
