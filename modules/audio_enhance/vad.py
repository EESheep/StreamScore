"""Silero VAD 语音活动检测。模型对象由调用方注入。"""
import logging

logger = logging.getLogger(__name__)


def detect_speech(vad_model, vad_utils, audio_path, threshold=0.5):
    """返回有声段时间区间列表 [{'start': float, 'end': float}, ...]"""
    _, _, read_audio, _, _ = vad_utils
    get_speech_timestamps = vad_utils[0]

    wav = read_audio(audio_path, sampling_rate=16000)
    timestamps = get_speech_timestamps(
        wav, vad_model,
        return_seconds=True,
        threshold=threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
    )
    logger.info("VAD found %d speech segments in %s", len(timestamps), audio_path)
    return timestamps
