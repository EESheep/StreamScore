"""阿里云 DashScope Paraformer ASR 模块。
通过 dashscope SDK 上传音频文件后调用 Paraformer 异步转写。
"""
import logging
import time
import requests

from modules.utils import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def run_asr_aliyun(audio_path, segments_path, output_path, config):
    """阿里云 Paraformer ASR 完整流程：上传 → 转写 → 过滤 → 保存。"""
    import dashscope
    from dashscope.audio.asr import Transcription

    cfg = config.get("asr_aliyun", {})
    dashscope.api_key = cfg.get("api_key", "")
    model = cfg.get("model", "paraformer-v2")
    language_hints = cfg.get("language_hints", ["zh", "en"])

    # ==================== 1. 上传音频文件 ====================
    logger.info("Uploading audio file: %s", audio_path)
    from dashscope.common.constants import FilePurpose
    upload_result = dashscope.Files.upload(
        file_path=audio_path,
        purpose=FilePurpose.assistants,
    )
    file_id = upload_result.output["uploaded_files"][0]["file_id"]
    logger.info("Uploaded file_id: %s", file_id)

    # 获取公网 URL
    file_res = dashscope.Files.get(file_id)
    file_url = file_res.output["url"]
    logger.info("File URL: %s", file_url)

    # ==================== 2. 提交转写任务 ====================
    logger.info("Submitting ASR task (model=%s)...", model)
    task_response = Transcription.async_call(
        model=model,
        file_urls=[file_url],
        language_hints=language_hints,
        **{k: v for k, v in cfg.get("extra_params", {}).items()},
    )
    task_id = task_response.output.task_id
    logger.info("Task ID: %s", task_id)

    # ==================== 3. 等待完成 ====================
    logger.info("Waiting for ASR to complete...")
    transcribe_response = Transcription.wait(task=task_id)
    if transcribe_response.status_code != 200:
        raise RuntimeError(
            f"ASR task failed: {transcribe_response.status_code} {transcribe_response.message}"
        )

    # ==================== 4. 下载结果 ====================
    result = transcribe_response.output["results"][0]
    if result["subtask_status"] != "SUCCEEDED":
        raise RuntimeError(f"ASR subtask failed: {result}")

    transcription_url = result["transcription_url"]
    resp = requests.get(transcription_url)
    data = resp.json()

    # ==================== 5. 转换为统一格式 ====================
    # Paraformer 返回格式: transcripts[].sentences[] 有 text, begin_time(ms), end_time(ms)
    raw_sentences = []
    for transcript in data.get("transcripts", []):
        raw_sentences.extend(transcript.get("sentences", []))

    # 去重合并：Paraformer 会将同一句话拆成多个极短且文本相同的片段
    merged = []
    for sent in raw_sentences:
        text = sent.get("text", "")
        begin_ms = sent.get("begin_time", 0)
        end_ms = sent.get("end_time", 0)
        if merged and merged[-1]["text"] == text:
            # 相同文本 → 扩展前一区间
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], end_ms)
        else:
            merged.append({"text": text, "begin_ms": begin_ms, "end_ms": end_ms})

    # 合并重叠的相邻段（间隔 < 0.3s 的相邻段且文本不同 → 拼接）
    merged2 = []
    for m in merged:
        if merged2 and m["begin_ms"] - merged2[-1]["end_ms"] < 300:
            merged2[-1]["text"] += m["text"]
            merged2[-1]["end_ms"] = max(merged2[-1]["end_ms"], m["end_ms"])
        else:
            merged2.append(m)

    words = []
    for m in merged2:
        words.append({
            "word": m["text"],
            "start": round(m["begin_ms"] / 1000, 2),
            "end": round(m["end_ms"] / 1000, 2),
            "probability": 1.0,
        })
    logger.info("Aliyun raw sentences: %d, after dedup+merge: %d",
                len(raw_sentences), len(words))

    logger.info("Aliyun ASR: %d sentences transcribed.", len(words))

    # ==================== 6. 过滤 + 保存 ====================
    from modules.asr import filter_host_transcript
    host_segments = read_jsonl(segments_path)
    transcript = filter_host_transcript(words, host_segments)
    write_jsonl(output_path, transcript)
    logger.info("Filtered to %d transcript segments.", len(transcript))
    return transcript
