"""音频增强子包纯工具函数。无状态，无模型依赖。"""
import soundfile as sf
import torch


def load_audio_segment(file_path, start, end, sr=16000):
    """加载音频文件的指定区间为波形 tensor，形状 (1, samples)。"""
    with sf.SoundFile(file_path) as f:
        f.seek(int(start * sr))
        frames = int((end - start) * sr)
        audio = f.read(frames, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
    return torch.from_numpy(audio).unsqueeze(0)


def get_audio_duration(file_path):
    """获取音频文件时长（秒）。"""
    info = sf.info(file_path)
    return info.duration


def load_full_audio(file_path, sr=16000):
    """加载整个音频文件为 numpy array。"""
    audio, _ = sf.read(file_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio
