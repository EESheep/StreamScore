# StreamScore 系统架构（v2.0）

**最后更新**：2026-05-30

---

## 全局流程图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 前置步骤 [FastAPI 轻量服务，pipeline 独立]                                      │
│   /api/v1/enroll/{room_id} → Demucs→VAD→Diarization → 候选 speaker 样本        │
│   操作者播放 + 波形图 → 确认主播 → 保存 ECAPA embedding 到 voiceprints/         │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ run_pipeline.py (主控)                                                         │
│                                                                              │
│  [1. 预处理]                                                                   │
│    ffmpeg 提取 audio_full.wav (16kHz 单声道, 损坏 FLV 尝试修复)                  │
│    XML 弹幕 → danmaku.jsonl (含时间归一化 + 丢弃日志)                            │
│                                                                              │
│  [2. 音频增强]  enhance_audio()                                                │
│    audio_full.wav                                                             │
│      │                                                                        │
│      ├─ chunking: ffmpeg 切 10min 块 + 30s overlap                            │
│      │                                                                        │
│      └─ 逐块循环 (VAD model + Pyannote pipeline 在循环外创建一次, 复用)          │
│          ├─ demucs → vocals_chunk_N.wav                                       │
│          ├─ vad → 有声区间 [{'start','end'}]                                   │
│          ├─ diarization → 块内 speaker 标签+时间区间                            │
│          ├─ voiceprint: 每 speaker 取最长段提取 ECAPA embedding                │
│          └─ 失败块: 跳过 + 记录 + 最终报告                                      │
│                                                                              │
│      └─ 跨块聚合:                                                              │
│          ├─ clustering: 贪心合并 embedding → 全局 speaker 身份                 │
│          ├─ voiceprint: 注册 embedding vs 聚类中心 → 锁定主播                   │
│          ├─ merge_gaps: 相邻主播段 ≤3s 且间隙无他人 → 合并                      │
│          ├─ 时间戳: chunk_offset + local_time → 全局时间                       │
│          └─ chunking: 5s crossfade 拼接 vocals_chunk_*.wav → vocals.wav      │
│                                                                              │
│    输出: host_speech_segments.jsonl + vocals.wav                              │
│                                                                              │
│  [3. ASR 转写]                                                                 │
│    faster-whisper large-v3 + int8 跑在 vocals.wav 上                           │
│    词级时间戳 → host_speech_segments 重叠过滤 → transcript.jsonl               │
│                                                                              │
│  [4. LLM 主题分段]                                                             │
│    segment_all(transcript) → 一次性全量送 DeepSeek                             │
│    输出 {start, end, title, summary}，立即拆为 JSONL                           │
│                                                                              │
│  [5. 填充 + 弹幕附着]                                                          │
│    fill_text(segments, transcript) → 整段包含, 可空 → 提示标记                  │
│    attach_danmaku(segments, danmaku) → 5s 缓冲, 去重, top 10, 密度/峰值       │
│                                                                              │
│  [6. 评分] (串行)                                                              │
│    score_segment() × N → 四个维度分(info/fun/interaction/emotion)              │
│    overall = sum(维度分 × config权重), 支持 per-room override                  │
│    API 失败: 重试 3 次指数退避, 仍失败则抛异常                                   │
│                                                                              │
│  [7. 剪辑建议 + 生成]                                                          │
│    generate_clip_suggestions() → LLM 输出 title/intro/tags (含原文+弹幕采样)    │
│    validate_clip_bounds() → 非法片段标记 needs_review                          │
│    generate_ffmpeg_commands() + execute_clips() → clip_*.mp4                  │
│                                                                              │
│  [输出]                                                                        │
│    highlights.jsonl + clip_*.mp4                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 数据文件一览

| 文件 | 格式 | 产生者 | 示例记录 |
|------|------|--------|---------|
| `audio_full.wav` | 16kHz mono PCM | 预处理 | — |
| `danmaku.jsonl` | JSONL | 预处理 | `{"ts":12.5, "text":"来了", "type":"scroll"}` |
| `vocals.wav` | 16kHz mono PCM | 音频增强 | — |
| `host_speech_segments.jsonl` | JSONL | 音频增强 | `{"start":60.0, "end":120.0}` |
| `transcript.jsonl` | JSONL | ASR | `{"start":60.0, "end":120.0, "text":"今天聊攻略", "confidence":0.92}` |
| `segments.jsonl` | JSONL | LLM 分段+填充+附着+评分 | `{"start":12.0, "end":350.0, "title":"新英雄教学", "summary":"...", "text":"...", "danmaku_sample":["111","来了"], "danmaku_count":847, "danmaku_density":8.5, "danmaku_peak":23, "scores":{"info":9,"fun":7,"interaction":8,"emotion":6}, "overall":7.5}` |
| `highlights.jsonl` | JSONL | LLM 剪辑 | `{"start":120.5, "end":345.0, "title":"...", "intro":"...", "tags":[...], "output":"clip_000.mp4", "status":"ok"}` |

---

## 目录结构

```
streamscore/
├── CONTEXT.md                              # 领域语言 + 架构决策
├── ARCHITECTURE.md                         # 本文档
├── config.yaml                             # 全局配置
├── requirements.txt                        # Python 依赖
├── run_pipeline.py                         # 主控脚本
├── modules/
│   ├── preprocess.py                       # 预处理 (ffmpeg 音频提取 + 弹幕解析)
│   ├── audio_enhance/                      # 音频增强子包
│   │   ├── __init__.py                     # 唯一编排者: enhance_audio()
│   │   ├── chunking.py                     # 切分 + crossfade 拼接
│   │   ├── demucs.py                       # Demucs CLI 人声分离
│   │   ├── vad.py                          # Silero VAD
│   │   ├── diarization.py                  # Pyannote 说话人分割
│   │   ├── voiceprint.py                   # ECAPA 声纹提取/匹配
│   │   ├── clustering.py                   # 跨块 embedding 贪心聚类
│   │   ├── merge_gaps.py                   # 间隙合并
│   │   └── utils.py                        # 纯函数工具 (无模型依赖)
│   ├── asr.py                              # ASR 转写
│   ├── llm_analysis.py                     # LLM 分段、评分、剪辑建议
│   ├── clip_generator.py                   # ffmpeg 命令生成 + 剪辑执行
│   └── utils.py                            # 全局工具函数
├── voiceprint_server/
│   ├── server.py                           # FastAPI 入口
│   ├── static/
│   │   └── index.html                      # 单页 UI (waveform + audio + 确认)
│   └── api/
│       └── enroll.py                       # /api/v1/enroll/* 路由实现
├── data/
│   └── voiceprints/
│       └── {room_id}/
│           ├── register.wav
│           ├── embedding.pt
│           └── meta.json
└── pretrained_models/
    └── spkrec/
```

---

## audio_enhance/ 子模块耦合约束

每个同级模块只依赖 `utils.py`（纯函数），模块之间零 import。只有 `__init__.py` 做编排。

| 模块 | 输入 | 输出 | 依赖 |
|------|------|------|------|
| `chunking.py` | `audio_full.wav`, 输出目录 | `[(chunk_path, offset)]`, `vocals.wav` | `utils.py` |
| `demucs.py` | 单个 wav 路径 | `vocals_chunk.wav` 路径 | 无外部模块 |
| `vad.py` | wav 路径 + VAD model 对象 | `[{'start','end'}]` | `utils.py` |
| `diarization.py` | wav 路径 + speech_intervals + pipeline 对象 | `pyannote Annotation` | 无外部模块 |
| `voiceprint.py` | wav 路径 + spkrec 对象 + 时间段 | `embedding tensor` | `utils.py` |
| `clustering.py` | `[{speaker_label, embedding, source_chunk}]` | `{global_id: [members]}` | 无外部模块 |
| `merge_gaps.py` | `[host_segments]`, `[diarization results]` | 合并后 `[host_segments]` | 无外部模块 |

---

## 声纹注册 API

| 方法 | 路径 | 功能 |
|------|------|------|
| `POST` | `/api/v1/enroll/{room_id}` | 启动注册处理 |
| `GET` | `/api/v1/enroll/{room_id}/status` | 查询状态 |
| `GET` | `/api/v1/enroll/{room_id}/candidates` | 候选 speaker 列表 + waveform 数据 |
| `GET` | `/api/v1/enroll/{room_id}/candidates/{id}/sample.wav` | 语音样本文件 |
| `POST` | `/api/v1/enroll/{room_id}/confirm` | `{"speaker_id": 1}` → 保存 embedding |
