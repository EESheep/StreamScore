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

## 测试
- 单模块调试: `python -m modules.preprocess --flv <path>`
- 目标硬件: NVIDIA RTX 3060 Laptop (8GB 显存)
- Python 3.10+, CUDA 11.8+
