"""LLM 分析模块：主题分段、评分、剪辑建议。"""
import json
import logging
import time
from collections import Counter

from modules.utils import read_jsonl, write_jsonl
from prompts.loader import render_prompt

logger = logging.getLogger(__name__)


def segment_all(transcript, deepseek_client, model="deepseek-v4-flash"):
    """一次性全量主题分段。返回 [{start, end, title, summary}, ...]"""
    # 传给 LLM 只保留 start/end/text
    snippet = [{"start": t["start"], "end": t["end"], "text": t["text"]} for t in transcript]
    prompt = render_prompt("segment.txt", transcript_json=json.dumps(snippet, ensure_ascii=False))

    logger.info("Running thematic segmentation (%d transcript entries)...", len(transcript))
    response = deepseek_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = _parse_json_response(response)
    if isinstance(result, dict) and "segments" in result:
        result = result["segments"]
    if not isinstance(result, list):
        raise ValueError(f"Unexpected segment_all response type: {type(result)}")
    logger.info("Segmented into %d thematic sections", len(result))
    return result


def fill_text(segments, transcript):
    """将原始 transcript 文本按整段包含填充到主题段，保留逐句时间戳。可空则标记。"""
    for seg in segments:
        seg_words = [
            t for t in transcript
            if t["start"] >= seg.get("start", 0) and t["end"] <= seg.get("end", 0)
        ]
        seg["text"] = " ".join(t["text"] for t in seg_words).strip()
        seg["transcript_entries"] = [
            {"start": t["start"], "text": t["text"]} for t in seg_words
        ]
        if not seg["text"]:
            seg["text"] = "（该时段主播未发言，可能存在过场/背景音乐/沉默）"
    return segments


def attach_danmaku(segments, danmaku_file, buffer_seconds=5, sample_top=10):
    """弹幕附着：去重 → 频次排序 → top N 采样，同时计算密度和峰值。"""
    danmakus = read_jsonl(danmaku_file)
    if not danmakus:
        for seg in segments:
            seg["danmaku_sample"] = []
            seg["danmaku_count"] = 0
            seg["danmaku_density"] = 0
            seg["danmaku_peak"] = 0
            seg["danmaku_peak_time"] = None
        return segments

    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0) + buffer_seconds

        # 收集区间内弹幕
        seg_danmakus = [d for d in danmakus if start <= d["ts"] <= end]

        # 去重 + 频次统计
        text_counts = Counter(d["text"] for d in seg_danmakus)
        # 按频次降序取 top N 条不同文本
        top_texts = [text for text, _ in text_counts.most_common(sample_top)]

        # 密度
        duration = max(end - start, 1)
        density = round(len(seg_danmakus) / duration, 2)

        # 峰值 (每秒弹幕数，取最密集的 1 秒)
        peak, peak_time = _compute_danmaku_peak(seg_danmakus, start, end)

        seg["danmaku_sample"] = top_texts
        seg["danmaku_count"] = len(seg_danmakus)
        seg["danmaku_density"] = density
        seg["danmaku_peak"] = peak
        seg["danmaku_peak_time"] = peak_time

    return segments


def score_segment(segment, deepseek_client, model="deepseek-v4-flash", max_retries=3):
    """评分单个段。返回四维度分，不含 overall。重试 3 次，指数退避。"""
    prompt = render_prompt("score.txt",
        title=segment.get("title", "未命名"),
        start=segment.get("start", 0),
        end=segment.get("end", 0),
        text=segment.get("text", ""),
        danmaku_sample=json.dumps(segment.get("danmaku_sample", []), ensure_ascii=False),
        danmaku_count=segment.get("danmaku_count", 0),
        danmaku_density=segment.get("danmaku_density", 0),
        danmaku_peak=segment.get("danmaku_peak", 0),
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = deepseek_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            scores = _parse_json_response(response)
            _validate_scores(scores)
            return scores
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("Score attempt %d/%d failed: %s. Retrying in %ds...",
                           attempt + 1, max_retries, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"Scoring failed after {max_retries} attempts. Last error: {last_error}")


def generate_clip_suggestions(high_scored_segments, deepseek_client, model="deepseek-v4-flash"):
    """生成剪辑建议+标注。包含 hook 候选、删除建议、保留核心。"""
    clips_input = []
    for seg in high_scored_segments:
        clips_input.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "title": seg.get("title"),
            "summary": seg.get("summary", ""),
            "transcript": seg.get("transcript_entries", []),
            "danmaku_sample": seg.get("danmaku_sample", []),
            "danmaku_count": seg.get("danmaku_count", 0),
            "danmaku_density": seg.get("danmaku_density", 0),
            "danmaku_peak": seg.get("danmaku_peak", 0),
            "danmaku_peak_time": seg.get("danmaku_peak_time"),
            "overall": seg.get("overall", 0),
            "scores": seg.get("scores", {}),
        })

    prompt = render_prompt("clip_suggest.txt", clips_json=json.dumps(clips_input, ensure_ascii=False))
    logger.info("Generating clip suggestions for %d segments...", len(clips_input))
    response = deepseek_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = _parse_json_response(response)
    if isinstance(result, dict) and "clips" in result:
        result = result["clips"]
    if not isinstance(result, list):
        raise ValueError(f"Unexpected clip_suggest response type: {type(result)}")
    return result


# ============================================================
# 内部辅助
# ============================================================

def _parse_json_response(response):
    """解析 LLM JSON 响应，处理常见格式问题。"""
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response content from LLM")
    # 尝试处理 markdown 代码块包裹
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(content)


def _validate_scores(scores):
    """校验评分结构：必须包含 info/fun/interaction/emotion 四个 1-10 的数值。"""
    required = ["info", "fun", "interaction", "emotion"]
    for key in required:
        if key not in scores:
            raise ValueError(f"Missing score dimension: {key}")
        val = scores[key]
        if not isinstance(val, (int, float)) or val < 1 or val > 10:
            raise ValueError(f"Invalid score for {key}: {val}")


def _compute_danmaku_peak(danmakus, seg_start, seg_end):
    """计算弹幕峰值和峰值时间。返回 (peak_count, peak_time)。"""
    if not danmakus:
        return 0, None
    duration = max(seg_end - seg_start, 1)
    bucket_count = int(duration) + 1
    buckets = [0] * bucket_count
    for d in danmakus:
        idx = int(d["ts"] - seg_start)
        if 0 <= idx < bucket_count:
            buckets[idx] += 1
    peak = max(buckets) if buckets else 0
    peak_idx = buckets.index(peak)
    peak_time = round(seg_start + peak_idx + 0.5, 1) if peak > 0 else None
    return peak, peak_time
