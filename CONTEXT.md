# StreamScore 领域语言与架构决策

**最后更新**：2026-05-30

---

## 领域词汇

| 术语 | 定义 |
|------|------|
| 主播 (Host) | 直播内容的主要产出者，系统目标提取其语音并评估其内容质量 |
| 声纹注册 (Voiceprint Enrollment) | 首次处理新主播时的人工前置步骤：从主播语音段中确认身份，提取 ECAPA embedding 缓存为 `data/voiceprints/{room_id}/` |
| 音源分离 (Source Separation) | Demucs htdemucs_ft 从原始音频中分离 vocals 和伴奏两部分 |
| 说话人分割 (Diarization) | Pyannote 3.1 在音频中区分"谁在什么时候说话"，分配块内临时 speaker 标签 |
| 声纹匹配 (Voiceprint Matching) | 用注册的 host embedding 对聚类后的 speaker embeddings 计算余弦相似度，锁定主播 |
| 跨块对齐 (Cross-chunk Speaker Alignment) | 对每块每个 speaker 提取 ECAPA embedding，贪心聚类统一多块的 speaker 身份 |
| 主题分段 (Thematic Segmentation) | LLM 根据转写文本识别话题切换边界，输出 {start, end, title, summary} |
| 弹幕附着 (Danmaku Attachment) | 在主题分段完成后，对每段按 [start, end+5s] 区间收集弹幕（top 10 采样 + 总量统计） |
| 间隙合并 (Gap Merge) | 相邻主播段间隔 ≤3s 且间隙内无其他 speaker 时合并为一段 |
| 分块处理 (Chunked Processing) | 长音频切分为 10min 块 + 30s 重叠，每块独立做 Demucs→VAD→Diarization，最后跨块对齐并拼接为全量 vocals.wav |

---

## 架构决策记录

### 决策 1：声纹获取方式
**选择**：人工标注（方案 C）。首次处理时人工确认主播身份，ECAPA embedding 缓存到 `data/voiceprints/{room_id}/`，后续复用。

### 决策 2：ASR 输入音频
**选择**：ASR 跑在 Demucs 输出的 `vocals.wav` 上。所有模块共用同一个时间基准。

### 决策 3：长音频处理策略
**选择**：分块处理。10min 块 + 30s 重叠，全局 embedding 聚类解决跨块 speaker 对齐，crossfade 拼接 vocals。

### 决策 4：GPU 资源管理
**选择**：每个函数内部管理自己创建的模型，调用方不需要知道内部细节。用 `del` + `torch.cuda.empty_cache()` 在函数内清理。

### 决策 5：LLM 分段策略
**选择**：一次性全量送入 DeepSeek（利用 1M 上下文），不做滑动窗口。

### 决策 6：弹幕附着时机与策略
**选择**：主题分段完成后统一附着。区间 = [seg.start, seg.end + 5s] 简单固定缓冲。删除 `merge_context`。

### 决策 7：切片填充逻辑位置
**选择**：`fill_text` 放在 `run_pipeline.py` 中，而非 LLM 调用函数内部。LLM 输出是纯分段，原文填充是机械步骤。

### 决策 8：弹幕采样
**选择**：传入 LLM 评分的弹幕取 top 10 + danmaku_count，避免无意义重复弹幕挤占 prompt。

### 决策 9：文件格式统一
**选择**：所有中间文件统一为 JSONL（每行一个 JSON 对象）。LLM 返回 JSON 数组后立即拆为 JSONL。

### 决策 10：删除无用字段
**选择**：去掉 `host_speech_segments` 中的 `confidence` 字段。去掉 config 中的 `window_size`、`overlap`、`beam_size`。

### 决策 11：time_segment 加载方式
**选择**：用 `soundfile.SoundFile.seek()` + `read()` 实现，不用 librosa。

### 决策 12：依赖清理
**选择**：删除 `openai-whisper` 和 `silero-vad` pip 依赖。VAD 通过 `torch.hub.load('snakers4/silero-vad', ...)` 加载。

### 决策 13：LLM 模型
**选择**：`deepseek-v4-flash`。1M 上下文，支持 JSON mode，成本极低（约 $0.015/场）。

### 决策 14：评分权重分离
**选择**：LLM 只输出四个维度分（info/fun/interaction/emotion），不输出 overall。overall 由 `run_pipeline.py` 用 `config.yaml` 中可配置的权重计算。权重可按主播类型调整。

### 决策 15：弹幕去重
**选择**：精确文本匹配去重，按出现频次降序排列。先去重再取 top 10。

### 决策 16：ASR 词边界过滤
**选择**：重叠即包含（`w.start < seg_end and w.end > seg_start`），不要求词完全落入区间，避免边界词丢失。

### 决策 17：声纹匹配取最长段
**选择**：块内 speaker embedding 取该 speaker 最长一段语音，与跨块聚类时的锚点策略一致。

### 决策 18：输出目录结构
**选择**：`chunks/` 和 `separated/` 子目录存放中间产物。每次运行开始时清理上一次的临时目录，保留当前运行中间文件直至下次覆盖。

### 决策 19：声纹注册交互
**选择**：FastAPI 轻量 Web 服务，单页面含波形图 + 音频播放，操作者选择对应 speaker 后保存 embedding，服务关闭。不作为常驻服务。

### 决策 20：Crossfade 策略
**选择**：短 crossfade（5 秒）消除拼接伪影。潜在风险：两块分离质量差异大时可能产生可感知音色跳变。定位参数：`audio_enhance.py` 中 `concat_vocals_with_crossfade` 的 `crossfade_duration`。

### 决策 21：分块处理模型复用
**选择**：`enhance_audio` 内部的 VAD 模型和 Pyannote pipeline 在分块循环前创建一次，所有块复用，循环结束后统一卸载。

### 决策 22：剪辑校验失败处理
**选择**：标记为 `needs_review`，跳过自动剪辑，在 highlights.jsonl 中标注状态。校验条件：start < 0 / end > duration / start >= end / 时长 < 3s / 时长 > 600s。

### 决策 23：弹幕解析增强
**选择**：丢弃弹幕时记录日志（含丢弃原因）。时间值归一化：> 10000 判为毫秒，除 1000 转秒。

### 决策 24：Diarization 单 speaker 场景
**选择**：仍执行声纹匹配，验证该 speaker 确为主播（防止录播或播放他人录音场景）。

### 决策 25：分块处理失败策略
**选择**：Demucs 崩溃或 Pyannote 失败 → 跳过该块 + 记录日志 + 最终报告失败块数量和原因。不做降级处理。

### 决策 26：LLM 评分并发
**选择**：串行调用。简单可靠，避免 rate limit 问题。

### 决策 27：剪辑建议输入精简
**选择**：保留原文和弹幕采样，但弹幕只传去重后的 top 10 + count。不传全量弹幕列表。

### 决策 28：模块结构设计原则
**选择**：每个功能领域独立为单文件模块。同级模块之间零 import，只通过 `__init__.py` 编排。模型对象由编排层创建并注入子模块。Claude Code 每次修改只触及一个文件。参见 `ARCHITECTURE.md` 中模块耦合约束表。
