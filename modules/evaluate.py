"""ASR 转写质量评估模块。
支持三种评估维度：
  1. 统计指标（句长分布、重复率、语速）
  2. 交叉对比（两个 ASR 输出的时间对齐差异分析）
  3. LLM 可读性评分（通过 DeepSeek 评估是否通顺）

用法:
  python -m modules.evaluate <transcript_a.jsonl> [transcript_b.jsonl] [--llm]
"""
import argparse
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


# ============================================================
# 统计指标
# ============================================================

def compute_stats(transcript):
    """计算单份转录的统计指标。"""
    if not transcript:
        return {}

    durations = [seg["end"] - seg["start"] for seg in transcript]
    texts = [seg["text"] for seg in transcript]
    total_dur = transcript[-1]["end"] - transcript[0]["start"] if len(transcript) > 1 else 0

    # 句长分布
    ultra_short = sum(1 for d in durations if d < 0.5)  # <0.5s 极短
    short = sum(1 for d in durations if 0.5 <= d < 2.0)
    medium = sum(1 for d in durations if 2.0 <= d < 10.0)
    long = sum(1 for d in durations if d >= 10.0)

    # 重复率 — 相邻完全相同
    repeats = sum(1 for i in range(1, len(texts)) if texts[i] == texts[i - 1])

    # 语速 — 字数/秒（中文按字符数，英文按空格分词）
    total_chars = sum(len(t) for t in texts)
    char_rate = total_chars / total_dur if total_dur > 0 else 0

    # 空/极短内容占比
    empty_like = sum(1 for t in texts if len(t.strip()) < 3)

    return {
        "segment_count": len(transcript),
        "total_duration_s": round(total_dur, 1),
        "avg_duration_s": round(sum(durations) / len(durations), 2),
        "duration_dist": {
            "ultra_short_<0.5s": ultra_short,
            "short_0.5-2s": short,
            "medium_2-10s": medium,
            "long_>10s": long,
        },
        "repeat_rate": round(repeats / max(len(texts) - 1, 1), 3),
        "chars_per_second": round(char_rate, 1),
        "empty_ratio": round(empty_like / max(len(texts), 1), 3),
    }


def print_stats(stats, label=""):
    """可读地打印统计结果。"""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  总段数:       {stats['segment_count']}")
    print(f"  音频总时长:   {stats['total_duration_s']:.1f}s ({stats['total_duration_s']/60:.1f}min)")
    print(f"  平均段长:     {stats['avg_duration_s']:.1f}s")
    print(f"  段长分布:     <0.5s={stats['duration_dist']['ultra_short_<0.5s']}  "
          f"0.5-2s={stats['duration_dist']['short_0.5-2s']}  "
          f"2-10s={stats['duration_dist']['medium_2-10s']}  "
          f">10s={stats['duration_dist']['long_>10s']}")
    print(f"  相邻重复率:   {stats['repeat_rate']:.1%}")
    print(f"  语速:         {stats['chars_per_second']:.1f} 字/秒  (中文正常3-5, 英文正常10-15)")
    print(f"  极短内容比:   {stats['empty_ratio']:.1%}")


def score_from_stats(stats):
    """根据统计指标给出综合分 (0-100)。"""
    score = 100
    details = []

    # 极短段过多 → 扣分
    us_ratio = stats["duration_dist"]["ultra_short_<0.5s"] / max(stats["segment_count"], 1)
    if us_ratio > 0.3:
        ded = int((us_ratio - 0.3) * 100)
        score -= ded
        details.append(f"极短句过多 ({us_ratio:.0%}): -{ded}")

    # 重复率高 → 扣分
    if stats["repeat_rate"] > 0.1:
        ded = int((stats["repeat_rate"] - 0.1) * 200)
        score -= ded
        details.append(f"重复率偏高 ({stats['repeat_rate']:.0%}): -{ded}")

    # 语速异常 → 扣分
    if stats["chars_per_second"] < 1.0:
        details.append(f"语速极慢 ({stats['chars_per_second']:.1f}字/秒) — 可能大片沉默或漏识别")
    elif stats["chars_per_second"] > 12:
        ded = 10
        score -= ded
        details.append(f"语速过快 ({stats['chars_per_second']:.1f}字/秒): -{ded}")

    return max(0, min(100, score)), details


# ============================================================
# 交叉对比
# ============================================================

def cross_compare(transcript_a, transcript_b, label_a="ASR-A", label_b="ASR-B"):
    """时间对齐比较两份转录的分歧。"""
    print(f"\n{'='*60}")
    print(f"  交叉对比: {label_a} vs {label_b}")
    print(f"{'='*60}")

    # 按时间窗口分组比较
    window_s = 60  # 每分钟一组
    if not transcript_a or not transcript_b:
        print("  无法对比：缺少转录数据")
        return

    max_end = max(transcript_a[-1]["end"], transcript_b[-1]["end"])
    n_windows = int(max_end / window_s) + 1

    # 每个窗口计算字符差
    diff_windows = []
    for w in range(n_windows):
        t0 = w * window_s
        t1 = t0 + window_s
        text_a = " ".join(s["text"] for s in transcript_a
                          if s["start"] < t1 and s["end"] > t0)
        text_b = " ".join(s["text"] for s in transcript_b
                          if s["start"] < t1 and s["end"] > t0)
        len_a = len(text_a)
        len_b = len(text_b)
        if len_a + len_b > 0:
            ratio = abs(len_a - len_b) / max(len_a, len_b, 1)
            diff_windows.append((t0, len_a, len_b, ratio))

    if not diff_windows:
        print("  无可比数据")
        return

    avg_diff = sum(w[3] for w in diff_windows) / len(diff_windows)
    high_diff = [w for w in diff_windows if w[3] > 0.5]

    print(f"  覆盖窗口: {len(diff_windows)} (各 {window_s}s)")
    print(f"  平均长度差异比: {avg_diff:.1%}")
    print(f"  高差异窗口 (>50%): {len(high_diff)}/{len(diff_windows)}")

    # 显示差异最大的前5个窗口
    if high_diff:
        print(f"\n  差异最大的时段:")
        high_diff.sort(key=lambda w: -w[3])
        for t0, la, lb, ratio in high_diff[:5]:
            mins = int(t0 / 60)
            print(f"    [{mins:3d}min]  {label_a}={la}字  {label_b}={lb}字  diff={ratio:.0%}")


# ============================================================
# LLM 可读性评分
# ============================================================

def evaluate_readability_llm(transcript, client, model, sample_n=10):
    """使用 LLM 抽样评分转写可读性。返回平均分。"""
    from prompts.loader import render_prompt
    import random

    # 只评长度 >2s 的段，更有意义
    candidates = [s for s in transcript if s["end"] - s["start"] > 2.0]
    if len(candidates) > sample_n:
        samples = random.sample(candidates, sample_n)
    else:
        samples = candidates[:sample_n]

    print(f"\n{'='*60}")
    print(f"  LLM 可读性评估（抽样 {len(samples)} 段）")
    print(f"{'='*60}")

    scores = []
    for i, seg in enumerate(samples):
        prompt = render_prompt("evaluate.txt", transcript=seg["text"])
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            scores.append({
                "start": seg["start"],
                "text": seg["text"][:60] + ("..." if len(seg["text"]) > 60 else ""),
                "readability": result.get("readability", 0),
                "issues": result.get("issues", []),
            })
            print(f"  [{i+1}/{len(samples)}] score={scores[-1]['readability']} "
                  f"\"{scores[-1]['text']}\"")
        except Exception as e:
            logger.warning("LLM evaluate failed: %s", e)
            scores.append({"start": seg["start"], "text": seg["text"][:60],
                           "readability": -1, "issues": [str(e)]})

    if scores:
        valid = [s["readability"] for s in scores if s["readability"] >= 0]
        avg = sum(valid) / len(valid) if valid else 0
        good = sum(1 for s in valid if s >= 7)
        bad = sum(1 for s in valid if s <= 3)
        print(f"\n  平均可读性: {avg:.1f}/10")
        print(f"  优质段(≥7): {good}/{len(valid)}")
        print(f"  劣质段(≤3): {bad}/{len(valid)}")

        # 展示问题样例
        issues_list = [s for s in scores if s["issues"] and len(s["issues"]) > 0]
        if issues_list:
            print(f"\n  有问题的段 ({len(issues_list)}):")
            for s in issues_list[:5]:
                print(f"    [{s['start']:.0f}s] score={s['readability']} issues={s['issues']}")

        return avg, scores
    return None, []


# ============================================================
# 主入口
# ============================================================

def load_transcript(path):
    """加载 JSONL 转录文件。"""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def main():
    parser = argparse.ArgumentParser(description="ASR 转录质量评估")
    parser.add_argument("transcript_a", help="主转录文件路径 (JSONL)")
    parser.add_argument("transcript_b", nargs="?", help="对比转录文件路径 (JSONL, 可选)")
    parser.add_argument("--label-a", default="ASR-A", help="主转录标签")
    parser.add_argument("--label-b", default="ASR-B", help="对比转录标签")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 可读性评分")
    parser.add_argument("--sample", type=int, default=10, help="LLM 评估抽样数量")
    parser.add_argument("--config", default="config.yaml", help="pipeline config 路径")
    parser.add_argument("--output", "-o", help="输出评估结果 JSON 路径")
    args = parser.parse_args()

    from modules.utils import setup_logging, load_config
    setup_logging()
    config = load_config(args.config)

    transcript_a = load_transcript(args.transcript_a)
    transcript_b = load_transcript(args.transcript_b) if args.transcript_b else []

    # ==================== 统计指标 ====================
    stats_a = compute_stats(transcript_a)
    score_a, detail_a = score_from_stats(stats_a)
    print_stats(stats_a, args.label_a)
    print(f"  综合分: {score_a}/100")
    for d in detail_a:
        print(f"    - {d}")

    if transcript_b:
        stats_b = compute_stats(transcript_b)
        score_b, detail_b = score_from_stats(stats_b)
        print_stats(stats_b, args.label_b)
        print(f"  综合分: {score_b}/100")
        for d in detail_b:
            print(f"    - {d}")

        # ==================== 交叉对比 ====================
        cross_compare(transcript_a, transcript_b, args.label_a, args.label_b)

    # ==================== LLM 评估 ====================
    llm_result = None
    if args.llm:
        ds_cfg = config.get("deepseek", {})
        from openai import OpenAI
        client = OpenAI(
            api_key=ds_cfg.get("api_key", ""),
            base_url=ds_cfg.get("base_url", "https://api.deepseek.com"),
        )
        model = ds_cfg.get("model", "deepseek-v4-flash")

        llm_result = {"a": None, "b": None}
        print(f"\n{'='*60}")
        print(f"  LLM 评估: {args.label_a}")
        print(f"{'='*60}")
        score_llm_a, detail_a_llm = evaluate_readability_llm(
            transcript_a, client, model, args.sample
        )

        if transcript_b:
            print(f"\n{'='*60}")
            print(f"  LLM 评估: {args.label_b}")
            print(f"{'='*60}")
            score_llm_b, detail_b_llm = evaluate_readability_llm(
                transcript_b, client, model, args.sample
            )
            llm_result = {"a": detail_a_llm, "b": detail_b_llm}

    # ==================== 综合报告 ====================
    print(f"\n{'='*60}")
    print(f"  综合评估报告")
    print(f"{'='*60}")
    report = {
        args.label_a: {"stat_score": score_a, "stats": stats_a},
    }
    if transcript_b:
        report[args.label_b] = {"stat_score": score_b, "stats": stats_b}

    if args.llm:
        if score_llm_a is not None:
            report[args.label_a]["llm_readability"] = round(score_llm_a, 1)
        if transcript_b and score_llm_b is not None:
            report[args.label_b]["llm_readability"] = round(score_llm_b, 1)

    for name, r in report.items():
        stat_s = r["stat_score"]
        llm_s = r.get("llm_readability", "N/A")
        print(f"  {name}: 统计分={stat_s}/100, LLM可读性={llm_s}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  报告已保存: {args.output}")

    print(f"\n{'='*60}")
    print(f"  评估完成")


if __name__ == "__main__":
    main()
