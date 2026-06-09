"""音频切分与 crossfade 拼接。纯 ffmpeg + numpy，无模型依赖。"""
import logging
import os
import subprocess
import numpy as np
import soundfile as sf

from modules.audio_enhance.utils import get_audio_duration

logger = logging.getLogger(__name__)


def split_audio(audio_path, output_dir, chunk_duration=600, overlap=30, sr=16000):
    """切分音频为带重叠的块，返回 [(chunk_path, offset_seconds), ...]"""
    duration = get_audio_duration(audio_path)
    chunks = []
    pos = 0
    idx = 0

    while pos < duration:
        chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")
        piece_duration = chunk_duration + overlap
        if pos + chunk_duration >= duration:
            piece_duration = duration - pos

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(pos),
            "-i", audio_path,
            "-t", str(piece_duration),
            "-ar", str(sr), "-ac", "1", "-f", "wav",
            chunk_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        chunks.append((chunk_path, pos))
        pos += chunk_duration
        idx += 1

    logger.info("Audio split into %d chunks (%ds each, %ds overlap)", idx, chunk_duration, overlap)
    return chunks


def concat_vocals_with_crossfade(vocals_chunk_paths, chunk_duration=600, overlap=30,
                                  crossfade_duration=5, sr=16000):
    """将所有块的 vocals 用 crossfade 拼接为完整波形。返回 numpy array。"""
    raw_arrays = [sf.read(path, dtype="float32")[0] for path in vocals_chunk_paths]
    # 统一转为单声道：Demucs 输出可能混有立体声/单声道
    arrays = []
    for arr in raw_arrays:
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr.mean(axis=1)  # stereo → mono
        arrays.append(arr)

    # 第一块：只取非重叠部分
    result = arrays[0][:chunk_duration * sr]

    for i in range(1, len(arrays)):
        prev_tail_start = max(0, len(result) - crossfade_duration * sr)
        prev_tail = result[prev_tail_start:]

        curr_head_len = min(crossfade_duration * sr, len(arrays[i]))
        curr_head = arrays[i][:curr_head_len]

        # linear crossfade
        fade_len = min(len(prev_tail), len(curr_head))
        fade_out = np.linspace(1, 0, fade_len, dtype="float32")
        fade_in = np.linspace(0, 1, fade_len, dtype="float32")
        blended = prev_tail[:fade_len] * fade_out + curr_head[:fade_len] * fade_in

        # 拼接：去掉 prev_tail 部分 + blended + 当前块剩余
        result = np.concatenate([
            result[:prev_tail_start],
            blended,
            arrays[i][fade_len:],
        ])

    return result


def write_vocals_full(vocals_array, output_path, sr=16000):
    """将拼接后的 vocals 写入文件。"""
    sf.write(output_path, vocals_array, sr, subtype="PCM_16")
    logger.info("Full vocals written: %s (%.1fs)", output_path, len(vocals_array) / sr)
