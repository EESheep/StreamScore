"""声纹注册 API 路由。"""
import json
import logging
import os
import threading

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/enroll", tags=["enroll"])
_enroll_state = {}  # {room_id: {"status": "processing"|"ready"|"error", "data": ..., "error": ...}}


class ConfirmRequest(BaseModel):
    speaker_id: int


def get_enroll_state(room_id):
    return _enroll_state.get(room_id)


@router.post("/{room_id}")
def start_enrollment(room_id: str):
    """启动声纹注册处理。接收 audio_full.wav 路径。"""
    if room_id in _enroll_state and _enroll_state[room_id]["status"] == "processing":
        return {"status": "processing", "message": "Already in progress"}

    _enroll_state[room_id] = {"status": "processing", "data": None, "error": None}
    thread = threading.Thread(target=_run_enrollment, args=(room_id,), daemon=True)
    thread.start()
    return {"status": "processing", "message": "Enrollment started"}


@router.get("/{room_id}/status")
def get_status(room_id: str):
    state = _enroll_state.get(room_id)
    if not state:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"status": state["status"], "error": state.get("error")}


@router.get("/{room_id}/candidates")
def get_candidates(room_id: str):
    state = _enroll_state.get(room_id)
    if not state or state["status"] != "ready":
        raise HTTPException(status_code=404, detail="Not ready or not found")
    candidates = []
    for i, c in enumerate(state["data"]["candidates"]):
        candidates.append({
            "id": i,
            "duration": c["duration"],
            "sample_url": f"/api/v1/enroll/{room_id}/candidates/{i}/sample.wav",
            "waveform_url": f"/api/v1/enroll/{room_id}/candidates/{i}/waveform",
        })
    return {"candidates": candidates}


@router.get("/{room_id}/candidates/{candidate_id}/sample.wav")
def get_candidate_sample(room_id: str, candidate_id: int):
    from fastapi.responses import FileResponse
    state = _enroll_state.get(room_id)
    if not state or state["status"] != "ready":
        raise HTTPException(status_code=404, detail="Not ready")
    path = state["data"]["candidates"][candidate_id]["sample_path"]
    return FileResponse(path, media_type="audio/wav")


@router.get("/{room_id}/candidates/{candidate_id}/waveform")
def get_candidate_waveform(room_id: str, candidate_id: int):
    """返回波形数据 (peaks) 供前端渲染。"""
    state = _enroll_state.get(room_id)
    if not state or state["status"] != "ready":
        raise HTTPException(status_code=404, detail="Not ready")
    return {"peaks": state["data"]["candidates"][candidate_id]["peaks"]}


@router.post("/{room_id}/confirm")
def confirm_enrollment(room_id: str, req: ConfirmRequest):
    """确认主播 speaker ID，保存 embedding。"""
    state = _enroll_state.get(room_id)
    if not state or state["status"] != "ready":
        raise HTTPException(status_code=404, detail="Not ready")

    candidate = state["data"]["candidates"][req.speaker_id]
    embedding = candidate["embedding"]
    sample_path = candidate["sample_path"]

    # 保存到 voiceprints/{room_id}/
    vp_dir = os.path.join("data", "voiceprints", room_id)
    os.makedirs(vp_dir, exist_ok=True)

    import shutil
    dest_wav = os.path.join(vp_dir, "register.wav")
    shutil.copy(sample_path, dest_wav)

    torch.save(embedding, os.path.join(vp_dir, "embedding.pt"))

    meta = {
        "room_id": room_id,
        "speaker_id": req.speaker_id,
        "duration": candidate["duration"],
    }
    with open(os.path.join(vp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    # 清理临时
    import shutil
    tmp = state["data"].get("temp_dir")
    if tmp and os.path.isdir(tmp):
        shutil.rmtree(tmp)

    del _enroll_state[room_id]
    return {"status": "ok", "message": f"Voiceprint saved for room {room_id}"}


# ============================================================
# 内部：后台处理
# ============================================================

def _run_enrollment(room_id):
    """后台运行 Demucs → VAD → Pyannote，提取候选 speaker 信息。"""
    try:
        state = _enroll_state[room_id]

        # 优先使用环境变量 ENROLL_AUDIO_PATH，否则回退到默认路径
        audio_path = os.environ.get(
            "ENROLL_AUDIO_PATH",
            f"data/recordings/{room_id}/latest/audio_full.wav"
        )
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # 创建临时目录
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix=f"enroll_{room_id}_")
        state["data"] = {"temp_dir": tmp_dir}

        # 运行 Demucs
        logger.info("[Enroll %s] Running Demucs...", room_id)
        import subprocess
        sep_dir = os.path.join(tmp_dir, "separated")
        cmd = ["demucs", "--two-stems", "vocals", "--device", "cuda", "-o", sep_dir, audio_path]
        subprocess.run(cmd, check=True)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(sep_dir, "htdemucs", base, "vocals.wav")

        # VAD
        logger.info("[Enroll %s] Running VAD...", room_id)
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad"
        )
        _, _, read_audio, _, _ = vad_utils
        get_speech_ts = vad_utils[0]
        wav = read_audio(vocals_path, sampling_rate=16000)
        speech_intervals = get_speech_ts(
            wav, vad_model, return_seconds=True,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

        # Pyannote
        logger.info("[Enroll %s] Running Diarization...", room_id)
        from pyannote.audio import Pipeline
        from pyannote.core import Segment
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_TOKEN", ""),
        )
        segments = [Segment(s["start"], s["end"]) for s in speech_intervals]
        diarization = pipeline({"uri": "audio", "audio": vocals_path}, segments=segments)

        # ECAPA 提取每 speaker 最长段
        logger.info("[Enroll %s] Extracting speaker embeddings...", room_id)
        from speechbrain.pretrained import SpeakerRecognition
        spkrec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec",
        )

        speaker_segs = {}
        for turn, _, spk in diarization.itertracks(yield_label=True):
            speaker_segs.setdefault(spk, []).append((turn.start, turn.end))

        candidates = []
        samples_dir = os.path.join(tmp_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        for spk, segs in speaker_segs.items():
            longest = max(segs, key=lambda s: s[1] - s[0])
            start, end = longest

            # 提取最长 10 秒
            dur = min(end - start, 10.0)
            sample_path = os.path.join(samples_dir, f"{spk}_sample.wav")
            data, sr = sf.read(vocals_path, start=int(start * 16000),
                               stop=int((start + dur) * 16000), dtype="float32")

            # 写到临时 WAV
            import soundfile as sf_write
            sf_write.write(sample_path, data, 16000, subtype="PCM_16")

            # ECAPA embedding
            from modules.audio_enhance.utils import load_audio_segment as las
            audio_tensor = las(vocals_path, start, start + dur)
            emb = spkrec.encode_waveform(audio_tensor)

            # 波形 peaks（简化为下采样）
            peaks = _compute_peaks(data, bins=200)

            candidates.append({
                "speaker_label": spk,
                "duration": round(dur, 2),
                "sample_path": sample_path,
                "embedding": emb.detach().cpu(),
                "peaks": peaks,
            })

        del vad_model, pipeline, spkrec
        torch.cuda.empty_cache()

        state["data"]["candidates"] = candidates
        state["status"] = "ready"
        logger.info("[Enroll %s] Ready with %d candidates.", room_id, len(candidates))

    except Exception as e:
        logger.exception("[Enroll %s] Failed", room_id)
        _enroll_state[room_id]["status"] = "error"
        _enroll_state[room_id]["error"] = str(e)


def _compute_peaks(audio, bins=200):
    """计算波形 peaks 用于前端可视化。"""
    if len(audio) == 0:
        return [0] * bins
    chunk_size = max(1, len(audio) // bins)
    peaks = []
    for i in range(bins):
        chunk = audio[i * chunk_size: (i + 1) * chunk_size]
        peaks.append(round(float(np.max(np.abs(chunk))), 4))
    return peaks
