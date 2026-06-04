"""剪辑生成模块：校验、ffmpeg 命令生成、执行。"""
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def validate_clip_bounds(clips, video_duration, min_duration=3, max_duration=600):
    """校验剪辑时间区间。非法片段标记 status='needs_review'，合法片段标记 status='ok'。"""
    validated = []
    for clip in clips:
        start = clip.get("start", 0)
        end = clip.get("end", 0)
        issues = []

        if start < 0:
            issues.append("start < 0")
        if end > video_duration:
            issues.append(f"end > video_duration ({video_duration})")
        if start >= end:
            issues.append(f"start ({start}) >= end ({end})")
        duration = end - start
        if duration < min_duration:
            issues.append(f"duration ({duration:.1f}s) < min ({min_duration}s)")
        if duration > max_duration:
            issues.append(f"duration ({duration:.1f}s) > max ({max_duration}s)")

        if issues:
            clip["status"] = "needs_review"
            clip["issues"] = issues
            logger.warning("Clip [%.1f-%.1f] needs review: %s", start, end, "; ".join(issues))
        else:
            clip["status"] = "ok"
        validated.append(clip)

    ok_count = sum(1 for c in validated if c["status"] == "ok")
    review_count = sum(1 for c in validated if c["status"] == "needs_review")
    logger.info("Validation: %d ok, %d needs_review", ok_count, review_count)
    return validated


def generate_ffmpeg_commands(clips, flv_path, output_dir, crf=23, audio_bitrate="128k"):
    """生成 ffmpeg 裁剪命令列表。仅对 status='ok' 的片段生成命令。"""
    commands = []
    for i, clip in enumerate(clips):
        if clip.get("status") != "ok":
            clip["output"] = None
            clip["ffmpeg_cmd"] = None
            continue

        start = clip["start"]
        duration = clip["end"] - clip["start"]
        out_path = os.path.join(output_dir, f"clip_{i:03d}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", flv_path,
            "-t", str(duration),
            "-c:v", "libx264", "-crf", str(crf),
            "-c:a", "aac", "-b:a", audio_bitrate,
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]
        commands.append(cmd)
        clip["output"] = out_path
        clip["ffmpeg_cmd"] = " ".join(cmd)

    return commands


def execute_clips(commands):
    """执行 ffmpeg 命令列表。"""
    for i, cmd in enumerate(commands):
        logger.info("Executing clip %d/%d: %s", i + 1, len(commands), cmd[-1])
        subprocess.run(cmd, check=True)
    logger.info("All %d clips generated.", len(commands))
