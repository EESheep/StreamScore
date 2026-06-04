"""间隙合并：合并相邻主播段，但跳过间隙中有其他人说话的情况。纯算法，无模型依赖。"""
import logging

logger = logging.getLogger(__name__)


def merge_adjacent(host_segments, all_diarization_results, host_label, max_gap=3.0):
    """
    合并相邻主播段。
    - host_segments: [{"start": float, "end": float}, ...]  按 start 升序
    - all_diarization_results: [(diarization_annotation, chunk_offset), ...]
    - host_label: 全局统一后的主播 speaker label
    - max_gap: 最大合并间隔（秒）
    返回合并后的 host_segments。
    """
    if len(host_segments) <= 1:
        return host_segments

    sorted_segs = sorted(host_segments, key=lambda s: s["start"])
    merged = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        gap_start = merged[-1]["end"]
        gap_end = seg["start"]
        gap = gap_end - gap_start

        if gap > max_gap:
            merged.append(seg)
            continue

        if _has_other_speaker_in_gap(gap_start, gap_end, all_diarization_results, host_label):
            merged.append(seg)
        else:
            merged[-1]["end"] = seg["end"]

    skipped = len(host_segments) - len(merged)
    if skipped > 0:
        logger.info("Gap merge: %d segments skipped (kept separate due to other speakers)", skipped)
    return merged


def _has_other_speaker_in_gap(gap_start, gap_end, all_diarization_results, host_label):
    """检查间隙中是否有非主播 speaker 的语音。"""
    for diarization, chunk_offset in all_diarization_results:
        for turn, _, spk in diarization.itertracks(yield_label=True):
            global_start = chunk_offset + turn.start
            global_end = chunk_offset + turn.end
            if global_end <= gap_start:
                continue
            if global_start >= gap_end:
                break
            if spk != host_label:
                return True
    return False
