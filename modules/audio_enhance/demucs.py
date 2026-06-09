"""Demucs 人声分离 —— 通过 subprocess 调用 CLI。"""
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def separate_vocals(audio_path, output_dir, device="cuda"):
    """运行 demucs CLI，返回 vocals.wav 路径。"""
    cmd = [
        "demucs", "--two-stems", "vocals",
        "--device", device,
        "-o", output_dir,
        audio_path,
    ]
    logger.info("Running Demucs: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(output_dir, "htdemucs", base, "vocals.wav")
    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"Demucs output not found: {vocals_path}")
    logger.info("Demucs output: %s", vocals_path)
    return vocals_path
