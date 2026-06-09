"""ECAPA 声纹提取与匹配。spkrec 对象由调用方注入。"""
import logging
import torch.nn.functional as F

from modules.audio_enhance.utils import load_audio_segment

logger = logging.getLogger(__name__)


def extract_embedding(spkrec, audio_path, start, end):
    """从音频指定区间提取 speaker embedding，形状 (1, 192)。"""
    audio = load_audio_segment(audio_path, start, end)
    if audio is None:
        return None
    # SpeechBrain 0.5.16: encode_batch, 1.0+: encode_waveform
    if hasattr(spkrec, "encode_waveform"):
        emb = spkrec.encode_waveform(audio)
    else:
        emb = spkrec.encode_batch(audio)
    # 确保输出形状为 (1, 192): encode_batch 可能返回 (1, 1, 192)
    while emb.dim() > 2:
        emb = emb.squeeze(1)  # squeeze middle dims
    return emb


def match_host_label(spkrec, speaker_embeddings, register_emb, threshold=0.65):
    """从 {speaker_label: embedding} 中找到匹配注册声纹的 label。返回 (label, sim) 或 (None, 0)。"""
    for label, emb in speaker_embeddings.items():
        sim = F.cosine_similarity(register_emb, emb, dim=1).item()
        if sim > threshold:
            logger.info("Host matched: speaker=%s, similarity=%.3f", label, sim)
            return label, sim
    return None, None


def load_register_embedding(spkrec, register_path):
    """从注册音频文件提取 embedding。"""
    from modules.audio_enhance.utils import get_audio_duration
    dur = get_audio_duration(register_path)
    emb = extract_embedding(spkrec, register_path, 0, dur)
    return emb
