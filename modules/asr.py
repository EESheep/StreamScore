"""ASR 转写模块：faster-whisper 转写 + 主播语音段过滤。"""
import logging

from modules.utils import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def transcribe_full_audio(audio_path, model_size="large-v3", device="cuda",
                          compute_type="int8", beam_size=5, language=None):
    """转写整段音频，返回词级时间戳列表 [{word, start, end, probability}]."""
    from faster_whisper import WhisperModel

    # faster-whisper 1.x 只接受 "cuda" 不接受 "cuda:0"
    if ":" in device:
        device = device.split(":")[0]
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info("Transcribing: %s (model=%s, compute=%s, language=%s)",
                audio_path, model_size, compute_type, language or "auto")
    segments, info = model.transcribe(
        audio_path, beam_size=beam_size, word_timestamps=True, language=language
    )
    logger.info("Detected language: %s (%.2f)", info.language, info.language_probability)

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word,
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "probability": round(w.probability, 4),
                })

    del model
    import torch
    torch.cuda.empty_cache()
    logger.info("Extracted %d words", len(words))
    return words


def filter_host_transcript(words, host_segments):
    """根据主播语音段（重叠即包含）过滤并合并为句子。返回 transcript 列表。"""
    if not host_segments:
        logger.warning("No host segments provided, returning empty transcript.")
        return []

    transcript = []
    for seg in host_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_words = [
            w for w in words
            if w["start"] < seg_end and w["end"] > seg_start
        ]
        if not seg_words:
            continue

        text = "".join(w["word"] for w in seg_words).strip()
        if not text:
            continue

        avg_prob = sum(w["probability"] for w in seg_words) / len(seg_words)
        transcript.append({
            "start": seg_start,
            "end": seg_end,
            "text": text,
            "confidence": round(avg_prob, 4),
        })

    # 去重：相邻段文本相同或完全包含时合并
    transcript = _dedup_transcript(transcript)
    logger.info("Filtered to %d transcript segments", len(transcript))
    return transcript


def _dedup_transcript(transcript):
    """合并相邻的重复/包含文本段。"""
    if len(transcript) <= 1:
        return transcript

    merged = [transcript[0]]
    for cur in transcript[1:]:
        prev = merged[-1]
        prev_text = prev["text"]
        cur_text = cur["text"]
        # 文本相同 或 前一段包含当前段 或 当前段包含前一段
        if (cur_text == prev_text
                or cur_text in prev_text
                or prev_text in cur_text):
            # 合并：取最长时间区间和最长文本
            prev["start"] = min(prev["start"], cur["start"])
            prev["end"] = max(prev["end"], cur["end"])
            if len(cur_text) > len(prev_text):
                prev["text"] = cur_text
        else:
            merged.append(cur)
    return merged
    return transcript


def run_asr(audio_path, segments_path, output_path, config):
    """ASR 完整流程：转写 → 过滤 → 保存。"""
    cfg = config.get("asr", {})
    gpu = config.get("gpu", {})

    host_segments = read_jsonl(segments_path)
    words = transcribe_full_audio(
        audio_path,
        model_size=cfg.get("model_size", "large-v3"),
        device=gpu.get("device", "cuda"),
        compute_type=cfg.get("compute_type", "int8"),
        language=cfg.get("language"),
    )
    transcript = filter_host_transcript(words, host_segments)
    write_jsonl(output_path, transcript)
    return transcript
