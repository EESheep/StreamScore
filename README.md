# StreamScore

直播内容智能评估与自动剪辑系统。自动分析直播回放，识别高能片段，生成带 hook 开头、去除水分的短视频成片。

## 特性

- **全自动流水线**：从原始 FLV 到成品 mp4，一键完成
- **音频增强**：Demucs 音源分离 → VAD → 说话人分割 → 声纹匹配，精确提取主播语音
- **多维度评分**：LLM 从信息密度、趣味性、互动性、情绪价值四个维度打分
- **智能剪辑**：自动识别高能片段，生成 hook 开头 + 去水分的成品
- **弹幕感知**：弹幕密度、峰值、去重采样融入评分与剪辑决策
- **半自动工作流**：输出 Premiere Pro CSV 标记文件，支持手动精修
- **声纹注册**：首次使用通过 Web UI 确认主播身份，后续自动识别

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+
- NVIDIA GPU (8GB+ 显存)，测试设备 RTX 3060 Laptop

### 安装

```bash
git clone git@github.com:EESheep/StreamScore.git
cd StreamScore
pip install -r requirements.txt
bash scripts/setup.sh  # 初始化 git hooks
```

### 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入 DeepSeek API Key、HuggingFace Token 等
```

### 声纹注册（首次使用）

```bash
python -m voiceprint_server.server --port 8080
# 浏览器打开 http://localhost:8080
# 上传主播音频 → 确认身份 → 保存声纹
```

### 运行

```bash
# 完整流水线
python run_pipeline.py --room_id <id> --date <date>

# 调试模式（跳过增强和剪辑）
python run_pipeline.py --room_id <id> --date <date> --skip-enhance --skip-clip
```

### 输出

```
/data/processed/{room_id}/{date}/
├── audio_full.wav              # 原始音频
├── vocals.wav                  # 分离后的主播人声
├── transcript.jsonl            # ASR 转写
├── segments.jsonl              # 主题分段+评分
├── highlights.jsonl            # 高光剪辑建议（含 hook/删除标注）
├── silence.json                # 静音分析
├── clip_000.mp4                # 原始二刀流片段
├── clip_000_composed.mp4       # 带 hook + 去水分的成品
└── clip_000_markers.csv        # PR 标记文件
```

## 架构

```
原始 FLV + 弹幕 XML
  │
  ├─ 预处理：提取音频、解析弹幕
  ├─ 音频增强：Demucs → VAD → Diarization → 声纹匹配
  ├─ ASR 转写：faster-whisper / 阿里 Paraformer
  ├─ 主题分段：LLM 全量分段
  ├─ 弹幕附着 + 评分：四维度打分
  └─ 剪辑生成：筛选 → 标注 → 合成
       ├─ clip_000.mp4（二刀流）
       ├─ clip_000_composed.mp4（带 hook）
       └─ clip_000_markers.csv（PR 标记）
```

详细架构见 [ARCHITECTURE.md](ARCHITECTURE.md)，领域语言与决策记录见 [CONTEXT.md](CONTEXT.md)。

## 技术栈

| 模块 | 技术 |
|------|------|
| 音源分离 | Demucs (htdemucs_ft) |
| VAD | Silero VAD |
| 说话人分割 | Pyannote 3.1 |
| 声纹识别 | ECAPA-TDNN (SpeechBrain) |
| ASR | faster-whisper (large-v3) / 阿里 Paraformer |
| LLM | DeepSeek v4 Flash |
| 剪辑引擎 | FFmpeg |

## 开发

```bash
# 单模块调试
python -m modules.preprocess --flv <path>

# 运行测试
pytest tests/
```

开发指南见 [CLAUDE.md](CLAUDE.md)。

## 许可证

[Apache License 2.0](LICENSE)
