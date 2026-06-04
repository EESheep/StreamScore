"""Pyannote 说话人分割。pipeline 对象由调用方注入。"""
import logging
from pyannote.core import Segment

logger = logging.getLogger(__name__)


def diarize(pipeline, audio_path, speech_intervals):
    """对音频的指定区间执行说话人分割，返回 pyannote Annotation。"""
    segments = [Segment(seg["start"], seg["end"]) for seg in speech_intervals]
    logger.info("Diarizing %d segments in %s", len(segments), audio_path)
    diarization = pipeline({"uri": "audio", "audio": audio_path}, segments=segments)
    return diarization
