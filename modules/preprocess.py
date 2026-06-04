"""预处理模块：音频提取 + 弹幕解析。"""
import json
import logging
import os
import subprocess
import xml.etree.ElementTree as ET

from modules.utils import write_jsonl

logger = logging.getLogger(__name__)


def extract_audio(flv_path, output_wav_path):
    """从 FLV 提取 16kHz 单声道 WAV，损坏文件尝试修复。"""
    cmd = [
        "ffmpeg", "-y", "-i", flv_path,
        "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
        output_wav_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        logger.warning("ffmpeg extraction failed, attempting repair...")
        repair_cmd = [
            "ffmpeg", "-y", "-err_detect", "ignore_err",
            "-i", flv_path,
            "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
            output_wav_path
        ]
        subprocess.run(repair_cmd, check=True, capture_output=True, text=True)
    logger.info("Audio extracted to %s", output_wav_path)


def _normalize_time(raw):
    """弹幕时间归一化：> 10000 判为毫秒，除 1000 转秒。"""
    try:
        t = float(raw)
    except (ValueError, TypeError):
        return None
    if t > 10000:
        t = t / 1000.0
    return t


def parse_danmaku(xml_path, output_jsonl_path):
    """B站弹幕 XML → JSONL。时间归一化，丢弃记录写日志。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    danmakus = []
    discarded = 0

    for d in root.findall("d"):
        attrs = d.get("p", "").split(",")
        if len(attrs) < 1:
            logger.debug("Discarded danmaku: missing p attribute")
            discarded += 1
            continue

        ts = _normalize_time(attrs[0])
        if ts is None:
            logger.debug("Discarded danmaku: invalid time '%s'", attrs[0])
            discarded += 1
            continue

        text = d.text
        if text is None:
            logger.debug("Discarded danmaku at %.2fs: null text", ts)
            discarded += 1
            continue

        type_code = attrs[1] if len(attrs) >= 2 else "0"

        danmakus.append({
            "ts": ts,
            "text": text.strip(),
            "type": type_code,
        })

    danmakus.sort(key=lambda x: x["ts"])
    write_jsonl(output_jsonl_path, danmakus)
    logger.info("Extracted %d danmaku items (%d discarded).", len(danmakus), discarded)
