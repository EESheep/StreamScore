"""音频增强流水线编排。唯一有权 import 同级模块的文件。"""
import json
import logging
import os
import shutil
import torch

# --- Windows 兼容：speechbrain fetch() 尝试 symlink，Windows 无权限则回退为 copy ---
import pathlib as _pl
_pl_symlink_original = _pl.Path.symlink_to
def _pl_symlink_patched(self, target, target_is_directory=False):
    try:
        _pl_symlink_original(self, target, target_is_directory)
    except OSError:
        shutil.copy2(str(target), str(self))
_pl.Path.symlink_to = _pl_symlink_patched

from modules.utils import write_jsonl
from modules.audio_enhance import chunking, demucs, vad, diarization, voiceprint, clustering, merge_gaps

logger = logging.getLogger(__name__)


def enhance_audio(config, wav_path, output_dir):
    """
    完整音频增强流水线。
    输入: audio_full.wav 路径, 全局 config, 输出目录
    输出: host_speech_segments 列表 (全局时间戳) + 完整 vocals.wav 路径
    """
    chunks_dir = os.path.join(output_dir, "chunks")
    separated_dir = os.path.join(output_dir, "separated")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(separated_dir, exist_ok=True)

    cfg_audio = config.get("audio", {})
    cfg_enhance = config.get("enhance", {})
    chunk_duration = cfg_audio.get("chunk_duration", 600)
    chunk_overlap = cfg_audio.get("chunk_overlap", 30)
    crossfade_duration = cfg_audio.get("crossfade_duration", 5)
    vad_threshold = cfg_enhance.get("vad_threshold", 0.5)
    vp_threshold = cfg_enhance.get("voiceprint_threshold", 0.65)
    merge_gap = cfg_enhance.get("merge_max_gap", 3.0)
    device = config.get("gpu", {}).get("device", "cuda:0")

    # ==================== 1. 加载声纹注册 ====================
    register_dir = config.get("voiceprint", {}).get("register_dir", "./data/voiceprints/")
    room_id = _extract_room_id(output_dir)
    register_emb_path = os.path.join(register_dir, room_id, "embedding.pt")
    register_audio_path = os.path.join(register_dir, room_id, "register.wav")

    spkrec = _load_spkrec(device)
    if os.path.exists(register_emb_path):
        register_emb = torch.load(register_emb_path, map_location=device)
        logger.info("Loaded cached voiceprint for room %s", room_id)
    elif os.path.exists(register_audio_path):
        register_emb = voiceprint.load_register_embedding(spkrec, register_audio_path)
    else:
        register_emb = None
        logger.warning(
            "No voiceprint found for room %s. "
            "Will select speaker with most speech time as host. "
            "Run voiceprint_server first to register host voiceprint.", room_id
        )

    # ==================== 2. 创建 VAD + Pyannote 模型 ====================
    vad_model, vad_utils = _load_vad(device)
    pipeline = _load_pyannote(config.get("hf_token", ""), device)

    # ==================== 3. 切分 ====================
    chunk_info = chunking.split_audio(wav_path, chunks_dir, chunk_duration, chunk_overlap)
    logger.info("Processing %d chunks...", len(chunk_info))

    # ==================== 4. 逐块处理 ====================
    all_embeddings = []       # [(spk_label, emb, chunk_idx), ...]
    all_diarization = {}      # {chunk_idx: (annotation, offset)} for merge_gaps
    vocals_chunks = []        # [(chunk_idx, vocals_path)]
    failed_blocks = []

    for idx, (chunk_path, offset) in enumerate(chunk_info):
        logger.info("--- Chunk %d/%d (offset=%.1fs) ---", idx + 1, len(chunk_info), offset)

        # 4a. Demucs
        try:
            vocals_path = demucs.separate_vocals(chunk_path, separated_dir)
        except Exception as e:
            logger.error("Chunk %d: Demucs failed: %s", idx, e)
            failed_blocks.append({"chunk": idx, "offset": offset, "stage": "demucs", "error": str(e)})
            continue
        vocals_chunks.append((idx, vocals_path))

        # 4b. VAD
        speech_intervals = vad.detect_speech(vad_model, vad_utils, vocals_path, vad_threshold)
        if not speech_intervals:
            logger.warning("Chunk %d: no speech detected, skipping", idx)
            failed_blocks.append({"chunk": idx, "offset": offset, "stage": "vad", "error": "no speech"})
            continue

        # 4c. Diarization
        try:
            diar = diarization.diarize(pipeline, vocals_path, speech_intervals)
        except Exception as e:
            logger.error("Chunk %d: Diarization failed: %s", idx, e)
            failed_blocks.append({"chunk": idx, "offset": offset, "stage": "diarization", "error": str(e)})
            continue
        all_diarization[idx] = (diar, offset)

        # 4d. 每 speaker 取最长段提取 embedding
        speaker_segments = {}
        for turn, _, spk in diar.itertracks(yield_label=True):
            speaker_segments.setdefault(spk, []).append((turn.start, turn.end))

        for spk, segs in speaker_segments.items():
            longest = max(segs, key=lambda s: s[1] - s[0])
            try:
                emb = voiceprint.extract_embedding(spkrec, vocals_path, longest[0], longest[1])
                if emb is not None:
                    all_embeddings.append((spk, emb, idx))
            except Exception as e:
                logger.warning("Chunk %d speaker %s: embedding extraction failed: %s", idx, spk, e)

    # ==================== 5. 跨块聚类 ====================
    if not all_embeddings:
        raise RuntimeError("No speaker embeddings extracted from any chunk.")
    global_clusters = clustering.cluster_speakers(all_embeddings)

    # ==================== 6. 声纹匹配 ====================
    # 每个聚类的均值 embedding
    cluster_embeddings = {}
    for gid, cluster_data in global_clusters.items():
        cluster_embeddings[gid] = clustering._compute_center(cluster_data["embeddings"])

    if register_emb is not None:
        host_label, _ = voiceprint.match_host_label(spkrec, cluster_embeddings, register_emb, vp_threshold)
    else:
        host_label = None

    if host_label is None:
        if len(global_clusters) == 1:
            host_label = list(global_clusters.keys())[0]
            logger.warning("Only one speaker found, assuming it's the host.")
        elif register_emb is None:
            # 无声纹注册时，选总说话时长最长的 speaker 作为主播
            total_dur = {}
            for gid, cluster_data in global_clusters.items():
                dur = 0.0
                for m in cluster_data["members"]:
                    cid = m["chunk_id"]
                    spk = m["speaker_label"]
                    if cid in all_diarization:
                        diar, _ = all_diarization[cid]
                        for turn, _, s in diar.itertracks(yield_label=True):
                            if s == spk:
                                dur += turn.end - turn.start
                total_dur[gid] = dur
            host_label = max(total_dur, key=total_dur.get)
            logger.info("No voiceprint: selected speaker %s (total %.1fs) as host from %d candidates",
                        host_label, total_dur[host_label], len(global_clusters))
        else:
            raise RuntimeError(
                f"No speaker matched host voiceprint. "
                f"Threshold={vp_threshold}, Found={len(global_clusters)} speakers."
            )

    # ==================== 7. 收集主播段 (全局时间戳) ====================
    host_segments = []
    for member in global_clusters[host_label]["members"]:
        chunk_idx = member["chunk_id"]
        chunk_spk_label = member["speaker_label"]
        if chunk_idx not in all_diarization:
            continue
        offset = chunk_info[chunk_idx][1]
        diar, _ = all_diarization[chunk_idx]

        for turn, _, spk in diar.itertracks(yield_label=True):
            if spk == chunk_spk_label:
                host_segments.append({
                    "start": round(offset + turn.start, 2),
                    "end": round(offset + turn.end, 2),
                })

    host_segments.sort(key=lambda s: s["start"])

    # ==================== 8. 间隙合并 ====================
    all_diar_list = list(all_diarization.values())
    host_segments = merge_gaps.merge_adjacent(
        host_segments, all_diar_list, host_label, merge_gap
    )

    # ==================== 9. 拼接 vocals ====================
    vocals_full_path = os.path.join(output_dir, "vocals.wav")
    vocals_paths_sorted = [p for _, p in sorted(vocals_chunks, key=lambda x: x[0])]
    vocals_np = chunking.concat_vocals_with_crossfade(
        vocals_paths_sorted, chunk_duration, chunk_overlap, crossfade_duration
    )
    chunking.write_vocals_full(vocals_np, vocals_full_path)

    # ==================== 10. 清理 ====================
    del spkrec, vad_model, pipeline
    torch.cuda.empty_cache()

    # 保存结果
    segments_file = os.path.join(output_dir, "host_speech_segments.jsonl")
    write_jsonl(segments_file, host_segments)

    # 失败报告
    if failed_blocks:
        logger.warning("=== %d chunk(s) failed ===", len(failed_blocks))
        for fb in failed_blocks:
            logger.warning("  Chunk %d (offset %ds) stage=%s: %s",
                           fb["chunk"], fb["offset"], fb["stage"], fb["error"])

    return host_segments


# ============================================================
# 内部模型加载函数
# ============================================================

def _load_spkrec(device):
    from speechbrain.pretrained import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec",
        run_opts={"device": device},
    )


def _load_vad(device):
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
    )
    # Silero VAD TorchScript 模型在 CUDA 上有 STFT buffer 设备不匹配问题，
    # 保持 CPU 推理，输入调用前手动将 wav 移到 CPU
    _ = device  # unused
    return model, utils


def _load_pyannote(hf_token, device):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if device.startswith("cuda"):
        pipeline = pipeline.to(torch.device(device))
    return pipeline


def _extract_room_id(output_dir):
    """从 output_dir 路径中提取 room_id（假设格式 .../room_id/date/）。"""
    parts = os.path.normpath(output_dir).split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"
