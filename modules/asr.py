"""ASR 转写模块：faster-whisper 转写 + 主播语音段过滤。"""
import logging

from modules.utils import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def transcribe_full_audio(audio_path, model_size="large-v3", device="cuda",
                          compute_type="int8", beam_size=5):
    """转写整段音频，返回词级时间戳列表 [{word, start, end, probability}]."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info("Transcribing: %s (model=%s, compute=%s)", audio_path, model_size, compute_type)
    segments, info = model.transcribe(audio_path, beam_size=beam_size, word_timestamps=True)
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

    logger.info("Filtered to %d transcript segments", len(transcript))
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
    )
    transcript = filter_host_transcript(words, host_segments)
    write_jsonl(output_path, transcript)
    return transcript
