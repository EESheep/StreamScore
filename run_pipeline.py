"""StreamScore 主控脚本。编排全流程六大阶段。"""
import argparse
import json
import logging
import os
import shutil
import sys

import torch
from openai import OpenAI

from modules.utils import setup_logging, load_config, read_jsonl, write_jsonl
from modules.preprocess import extract_audio, parse_danmaku
from modules.audio_enhance import enhance_audio
from modules.asr import run_asr
from modules.llm_analysis import (
    segment_all, fill_text, attach_danmaku,
    score_segment, generate_clip_suggestions,
)
from modules.clip_generator import (
    validate_clip_bounds, generate_ffmpeg_commands, execute_clips,
    detect_silence_gaps, generate_pr_markers_csv, compose_hook_clips,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="StreamScore - 直播内容智能评估与剪辑")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--room_id", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--skip-enhance", action="store_true", help="跳过音频增强")
    parser.add_argument("--skip-clip", action="store_true", help="跳过实际剪辑输出")
    args = parser.parse_args()

    # ==================== 初始化 ====================
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    setup_logging()
    config = load_config(args.config)

    flv_path = config["input"]["flv_template"].format(room_id=args.room_id, date=args.date)
    danmaku_xml = config["input"]["danmaku_template"].format(room_id=args.room_id, date=args.date)
    output_dir = config["input"]["output_dir"].format(room_id=args.room_id, date=args.date)
    os.makedirs(output_dir, exist_ok=True)

    # 清理上次运行的临时目录
    _cleanup_temp_dirs(output_dir)

    logger.info("=== StreamScore Pipeline ===")
    logger.info("Room: %s | Date: %s | Output: %s", args.room_id, args.date, output_dir)

    # ==================== 1. 预处理 ====================
    logger.info("--- Stage 1: Preprocessing ---")
    audio_path = os.path.join(output_dir, "audio_full.wav")
    extract_audio(flv_path, audio_path)

    danmaku_jsonl = os.path.join(output_dir, "danmaku.jsonl")
    parse_danmaku(danmaku_xml, danmaku_jsonl)

    # ==================== 2. 音频增强 ====================
    if args.skip_enhance:
        logger.info("--- Stage 2: Audio Enhancement (SKIPPED) ---")
    else:
        logger.info("--- Stage 2: Audio Enhancement ---")
        host_segments = enhance_audio(config, audio_path, output_dir)
        logger.info("Extracted %d host speech segments.", len(host_segments))

    host_segments_path = os.path.join(output_dir, "host_speech_segments.jsonl")
    host_segments = read_jsonl(host_segments_path)
    vocals_path = os.path.join(output_dir, "vocals.wav")

    # ==================== 3. ASR 转写 ====================
    logger.info("--- Stage 3: ASR Transcription ---")
    transcript_path = os.path.join(output_dir, "transcript.jsonl")
    asr_provider = config.get("asr_provider", "faster-whisper")
    if asr_provider == "aliyun":
        from modules.asr_aliyun import run_asr_aliyun
        # 阿里 Paraformer 用原始音频，避免 Demucs 伪影干扰
        transcript = run_asr_aliyun(audio_path, host_segments_path, transcript_path, config)
    else:
        transcript = run_asr(vocals_path, host_segments_path, transcript_path, config)
    logger.info("Transcribed %d segments.", len(transcript))

    # ==================== 4. LLM 初始化 ====================
    ds_cfg = config["deepseek"]
    deepseek_client = OpenAI(
        api_key=ds_cfg["api_key"],
        base_url=ds_cfg.get("base_url", "https://api.deepseek.com"),
    )
    model = ds_cfg.get("model", "deepseek-v4-flash")

    # ==================== 5. 主题分段 ====================
    logger.info("--- Stage 4: Thematic Segmentation ---")
    segments = segment_all(transcript, deepseek_client, model)
    segments_path = os.path.join(output_dir, "segments.jsonl")
    write_jsonl(segments_path, segments)
    logger.info("Segmented into %d thematic sections.", len(segments))

    # ==================== 6. 填充 + 弹幕附着 ====================
    logger.info("--- Stage 5: Fill Text + Attach Danmaku ---")
    segments = fill_text(segments, transcript)

    dm_cfg = config.get("danmaku", {})
    segments = attach_danmaku(
        segments, danmaku_jsonl,
        buffer_seconds=dm_cfg.get("buffer_seconds", 5),
        sample_top=dm_cfg.get("sample_top", 10),
    )

    # ==================== 7. 评分 ====================
    logger.info("--- Stage 6: Scoring ---")
    for i, seg in enumerate(segments):
        logger.info("Scoring segment %d/%d: %s", i + 1, len(segments), seg.get("title", "未命名"))
        scores = score_segment(seg, deepseek_client, model)
        seg["scores"] = scores

    # 计算 overall
    weights = _get_scoring_weights(config, args.room_id)
    for seg in segments:
        s = seg["scores"]
        seg["overall"] = round(sum(
            s[dim] * weights[dim] for dim in ["info", "fun", "interaction", "emotion"]
        ), 2)

    write_jsonl(segments_path, segments)
    logger.info("Scoring complete. Score range: %.2f - %.2f",
                min(s["overall"] for s in segments),
                max(s["overall"] for s in segments))

    # ==================== 8. 剪辑 ====================
    logger.info("--- Stage 7: Clip Generation ---")
    threshold = config.get("llm", {}).get("score_threshold", 7.0)
    high_scored = [seg for seg in segments if seg.get("overall", 0) >= threshold]
    logger.info("%d segments above threshold (%.1f)", len(high_scored), threshold)

    clips = generate_clip_suggestions(high_scored, deepseek_client, model)

    # VAD 静音检测 (用于 CSV 标记)
    silence_deletes, silence_transitions = detect_silence_gaps(
        vocals_path,
        min_delete_silence=2.0,
        min_transition_silence=4.0,
    )

    # 校验
    from modules.audio_enhance.utils import get_audio_duration
    video_duration = get_audio_duration(audio_path)
    clip_cfg = config.get("clip", {})
    clips = validate_clip_bounds(
        clips, video_duration,
        min_duration=clip_cfg.get("min_duration", 3),
        max_duration=clip_cfg.get("max_duration", 600),
    )

    highlights_path = os.path.join(output_dir, "highlights.jsonl")
    write_jsonl(highlights_path, clips)

    import json as _json
    silence_path = os.path.join(output_dir, "silence.json")
    with open(silence_path, "w", encoding="utf-8") as _f:
        _json.dump({
            "delete_suggestions": silence_deletes,
            "transition_points": silence_transitions,
        }, _f, ensure_ascii=False, indent=2)
    logger.info("Silence analysis saved: %d delete + %d transition",
                len(silence_deletes), len(silence_transitions))

    # 生成 PR CSV 标记
    silence_data = {"delete_suggestions": silence_deletes, "transition_points": silence_transitions}
    generate_pr_markers_csv(clips, segments, silence_data, output_dir)

    # 生成并执行剪辑命令
    ffmpeg_cmds = generate_ffmpeg_commands(
        clips, flv_path, output_dir,
        crf=clip_cfg.get("crf", 23),
        audio_bitrate=clip_cfg.get("audio_bitrate", "128k"),
    )

    if not args.skip_clip:
        ok_cmds = [cmd for cmd in ffmpeg_cmds if cmd is not None]
        execute_clips(ok_cmds)

        # 生成带 hook + 去水分的成品
        compose_hook_clips(
            clips, segments, silence_data, flv_path, output_dir,
            hook_duration=clip_cfg.get("hook_duration", 4.0),
            hook_before=clip_cfg.get("hook_before", 1.0),
            crf=clip_cfg.get("crf", 23),
            audio_bitrate=clip_cfg.get("audio_bitrate", "128k"),
        )
    else:
        logger.info("Clip execution skipped (--skip-clip).")

    # ==================== 完成 ====================
    torch.cuda.empty_cache()
    logger.info("=== Pipeline Complete ===")
    logger.info("Output: %s", output_dir)
    logger.info("Files: audio_full.wav, vocals.wav, danmaku.jsonl, "
                "host_speech_segments.jsonl, transcript.jsonl, segments.jsonl, highlights.jsonl")


# ============================================================
# 内部辅助
# ============================================================

def _get_scoring_weights(config, room_id):
    """返回评分权重：per-room override > 全局默认。"""
    default_weights = config.get("scoring_weights", {
        "info": 0.3, "fun": 0.25, "interaction": 0.25, "emotion": 0.2
    })
    overrides = config.get("room_overrides", {})
    if room_id in overrides:
        override_weights = overrides[room_id].get("scoring_weights")
        if override_weights:
            logger.info("Using per-room scoring weights for %s", room_id)
            return override_weights
    return default_weights


def _cleanup_temp_dirs(output_dir):
    """清理上次运行的临时目录。"""
    for subdir in ["chunks", "separated"]:
        path = os.path.join(output_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path)
            logger.info("Cleaned up %s", path)


if __name__ == "__main__":
    main()
