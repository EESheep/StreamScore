"""剪辑生成模块：校验、ffmpeg 命令生成、执行、VAD 静音检测、PR CSV 标记。"""
import csv
import logging
import os
import subprocess

import torch

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


def detect_silence_gaps(audio_path, min_delete_silence=2.0, min_transition_silence=4.0):
    """使用 Silero VAD 检测静音间隙。
    返回:
    - delete_suggestions: [{"start": s, "end": e, "duration": d}, ...]  >2s 沉默段
    - transition_points: [{"start": s, "end": e, "duration": d}, ...]  >4s 长停顿
    """
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
    )
    _, _, read_audio, _, _ = vad_utils
    get_speech_timestamps = vad_utils[0]

    wav = read_audio(audio_path, sampling_rate=16000)
    wav = wav.cpu()
    speech_segments = get_speech_timestamps(
        wav, vad_model,
        return_seconds=True,
    )

    if not speech_segments:
        logger.warning("VAD found no speech in %s", audio_path)
        del vad_model
        torch.cuda.empty_cache()
        return [], []

    # 从语音段间隙提取静音
    delete_suggestions = []
    transition_points = []

    for i in range(len(speech_segments) + 1):
        if i == 0:
            gap_start = 0.0
            gap_end = speech_segments[0]["start"]
        elif i == len(speech_segments):
            gap_start = speech_segments[-1]["end"]
            gap_end = None  # 片尾沉默暂不处理
        else:
            gap_start = speech_segments[i - 1]["end"]
            gap_end = speech_segments[i]["start"]

        if gap_end is None:
            continue

        duration = round(gap_end - gap_start, 1)
        if duration >= min_transition_silence:
            transition_points.append({
                "start": round(gap_start, 1),
                "end": round(gap_end, 1),
                "duration": duration,
            })
        elif duration >= min_delete_silence:
            delete_suggestions.append({
                "start": round(gap_start, 1),
                "end": round(gap_end, 1),
                "duration": duration,
            })

    del vad_model
    torch.cuda.empty_cache()
    logger.info("Silence detection: %d delete (>%.1fs), %d transition (>%.1fs)",
                len(delete_suggestions), min_delete_silence,
                len(transition_points), min_transition_silence)
    return delete_suggestions, transition_points


def generate_pr_markers_csv(clips, segments, silence_data, output_dir):
    """为每个 clip 生成 PR CSV 标记文件。
    产出: output_dir/clip_000_markers.csv, ...
    PR 导入方式: File > Import > Markers, 选择 CSV 文件
    """
    silence_deletes = silence_data.get("delete_suggestions", [])
    silence_transitions = silence_data.get("transition_points", [])

    for i, clip in enumerate(clips):
        if clip.get("status") != "ok":
            continue

        clip_start = clip["start"]
        clip_end = clip["end"]
        rows = []

        # ── Hook 候选: LLM 高分句 + 转折句 ──
        for hc in clip.get("hook_candidates", []):
            hc_time = hc.get("time", clip_start)
            label = "Hook-高分句" if hc["type"] == "highlight_sentence" else "Hook-转折句"
            rows.append({
                "Name": label,
                "In": hc_time,
                "Out": hc_time,
                "Duration": 0,
                "Comment": f"{hc.get('text', '')} | {hc.get('reason', '')}",
            })

        # ── Hook 候选: 弹幕峰 ──
        match_seg = _find_matching_segment(clip, segments)
        if match_seg and match_seg.get("danmaku_peak_time"):
            peak_t = match_seg["danmaku_peak_time"]
            if clip_start <= peak_t <= clip_end:
                rows.append({
                    "Name": "Hook-弹幕峰",
                    "In": peak_t,
                    "Out": peak_t,
                    "Duration": 0,
                    "Comment": f"弹幕峰值 {match_seg.get('danmaku_peak', 0)}条/秒",
                })

        # ── 建议删除: LLM 跑题 + 无效互动 ──
        for cs in clip.get("cut_suggestions", []):
            cs_start = cs.get("start", clip_start)
            cs_end = cs.get("end", clip_end)
            label = "建议删除-跑题" if cs["type"] == "off_topic" else "建议删除-无效互动"
            rows.append({
                "Name": label,
                "In": cs_start,
                "Out": cs_end,
                "Duration": round(cs_end - cs_start, 1),
                "Comment": cs.get("reason", ""),
            })

        # ── 建议删除: VAD 静音 ──
        for sg in silence_deletes:
            if sg["start"] < clip_end and sg["end"] > clip_start:
                rows.append({
                    "Name": "建议删除-静音",
                    "In": sg["start"],
                    "Out": sg["end"],
                    "Duration": sg["duration"],
                    "Comment": f"VAD检测 {sg['duration']}s沉默",
                })

        # ── 转场建议: Segment 边界 ──
        rows.append({
            "Name": "转场-片段开头",
            "In": clip_start,
            "Out": clip_start,
            "Duration": 0,
            "Comment": clip.get("title", ""),
        })
        if match_seg:
            rows.append({
                "Name": "转场-话题边界",
                "In": match_seg.get("end", clip_end),
                "Out": match_seg.get("end", clip_end),
                "Duration": 0,
                "Comment": f"话题结束: {match_seg.get('title', '')}",
            })

        # ── 转场建议: VAD 长停顿 ──
        for tp in silence_transitions:
            if tp["start"] < clip_end and tp["end"] > clip_start:
                rows.append({
                    "Name": "转场-长停顿",
                    "In": tp["start"],
                    "Out": tp["end"],
                    "Duration": tp["duration"],
                    "Comment": f"VAD检测 {tp['duration']}s长停顿",
                })

        # ── 保留核心 ──
        for kc in clip.get("keep_core", []):
            rows.append({
                "Name": "保留核心",
                "In": kc.get("start", clip_start),
                "Out": kc.get("end", clip_end),
                "Duration": round(kc.get("end", clip_end) - kc.get("start", clip_start), 1),
                "Comment": kc.get("reason", ""),
            })

        # ── 写入 CSV ──
        csv_path = os.path.join(output_dir, f"clip_{i:03d}_markers.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Name", "In", "Out", "Duration", "Comment"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Clip %d: %d markers → %s", i, len(rows), csv_path)


def _find_matching_segment(clip, segments):
    """通过时间重叠找到 clip 对应的原始 segment。"""
    clip_start = clip.get("start", 0)
    clip_end = clip.get("end", 0)
    best_overlap = 0
    best_seg = None
    for seg in segments:
        overlap = min(clip_end, seg.get("end", 0)) - max(clip_start, seg.get("start", 0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_seg = seg
    return best_seg


def compose_hook_clips(clips, segments, silence_data, flv_path, output_dir,
                       hook_duration=4.0, hook_before=1.0,
                       crf=23, audio_bitrate="128k"):
    """生成带 hook 开头 + 去水分的成品 mp4。
    每个 clip 先按删除建议切分保留段，再拼接 hook 作为开头。
    产出: output_dir/clip_000_composed.mp4, ...
    """
    silence_deletes = silence_data.get("delete_suggestions", [])
    temp_dir = os.path.join(output_dir, "_compose_temp")
    os.makedirs(temp_dir, exist_ok=True)

    composed_paths = []
    for i, clip in enumerate(clips):
        if clip.get("status") != "ok":
            composed_paths.append(None)
            continue

        clip_start = clip["start"]
        clip_end = clip["end"]
        concat_parts = []

        # ─ 1. 汇总所有删除区间 ─
        cuts = []
        for cs in clip.get("cut_suggestions", []):
            cuts.append((cs.get("start", clip_start), cs.get("end", clip_end)))
        for sg in silence_deletes:
            if sg["start"] < clip_end and sg["end"] > clip_start:
                cuts.append((sg["start"], sg["end"]))

        cuts = _merge_intervals(sorted(cuts))

        # ─ 2. 选取 hook ─
        hook_range = _select_hook(clip, segments, hook_before, hook_duration)
        if hook_range:
            h_start, h_end = hook_range
            # 避免 hook 内容在正文中重复
            cuts.append((h_start, h_end))
            cuts = _merge_intervals(sorted(cuts))
            concat_parts.append(("hook", h_start, h_end))

        # ─ 3. 计算保留区间（clip 范围减去所有 cuts）─
        keep_ranges = _invert_intervals(cuts, clip_start, clip_end)

        # ─ 4. 逐段提取 ─
        part_idx = 0
        for k_start, k_end in keep_ranges:
            if k_end - k_start < 0.5:
                continue
            concat_parts.append(("keep", k_start, k_end))
            part_idx += 1

        if not concat_parts:
            logger.warning("Clip %d: no content after cutting, skipped", i)
            composed_paths.append(None)
            continue

        if len(concat_parts) == 1:
            # 只有一段，直接单段提取
            _, s, e = concat_parts[0]
            out_path = os.path.join(output_dir, f"clip_{i:03d}_composed.mp4")
            _extract_segment(flv_path, s, e - s, out_path, crf, audio_bitrate)
            composed_paths.append(out_path)
            continue

        # ─ 5. 多段 concat ─
        part_files = []
        for pi, (ptype, ps, pe) in enumerate(concat_parts):
            part_path = os.path.join(temp_dir, f"clip{i:03d}_p{pi:02d}.mp4")
            _extract_segment(flv_path, ps, pe - ps, part_path, crf, audio_bitrate)
            part_files.append(part_path)

        concat_list_path = os.path.join(temp_dir, f"clip{i:03d}_concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for pf in part_files:
                f.write(f"file '{pf}'\n")

        out_path = os.path.join(output_dir, f"clip_{i:03d}_composed.mp4")
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            out_path,
        ]
        subprocess.run(concat_cmd, check=True)

        for pf in part_files:
            os.remove(pf)
        os.remove(concat_list_path)

        logger.info("Clip %d composed: %d parts → %s", i, len(concat_parts), out_path)
        composed_paths.append(out_path)

    # 清理临时目录
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
    return composed_paths


def _select_hook(clip, segments, hook_before, hook_duration):
    """从 clip 的 hook_candidates 中选取最佳 hook，返回 (start, end) 区间。"""
    hook_candidates = clip.get("hook_candidates", [])
    clip_start = clip["start"]
    clip_end = clip["end"]

    # 优先级: highlight_sentence > turning_point
    best_hook = None
    for hc in hook_candidates:
        if hc.get("type") == "highlight_sentence":
            best_hook = hc
            break
    if not best_hook and hook_candidates:
        best_hook = hook_candidates[0]
    if not best_hook:
        # fallback: 弹幕峰
        match_seg = _find_matching_segment(clip, segments)
        if match_seg and match_seg.get("danmaku_peak_time"):
            best_hook = {"time": match_seg["danmaku_peak_time"]}
    if not best_hook:
        return None

    hook_time = best_hook.get("time")
    if hook_time is None:
        return None
    h_start = max(clip_start, hook_time - hook_before)
    h_end = min(clip_end, h_start + hook_duration)
    if h_end - h_start < 1.0:
        return None
    return (h_start, h_end)


def _merge_intervals(intervals):
    """合并重叠/相邻的时间区间。"""
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _invert_intervals(cuts, clip_start, clip_end):
    """给定删除区间列表，返回保留区间列表。"""
    keep = []
    cursor = clip_start
    for cs, ce in cuts:
        cs = max(clip_start, min(cs, clip_end))
        ce = max(clip_start, min(ce, clip_end))
        if cs > cursor:
            keep.append((cursor, cs))
        cursor = max(cursor, ce)
    if cursor < clip_end:
        keep.append((cursor, clip_end))
    return keep


def _extract_segment(flv_path, start, duration, out_path, crf, audio_bitrate):
    """从 FLV 中提取一段，编码输出 mp4。"""
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
    subprocess.run(cmd, check=True, capture_output=True)
