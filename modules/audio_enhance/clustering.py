"""跨块 speaker embedding 贪心聚类。纯算法，无模型依赖。"""
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def cluster_speakers(all_embeddings, threshold=0.75):
    """
    输入: [(speaker_label, embedding_tensor, source_chunk_id), ...]
    输出: {global_speaker_id: [members], ...}
    贪心合并：每个 embedding 与现有聚类中心比较，相似则加入，否则新建聚类。
    """
    clusters = []  # list of {"label": str, "members": [], "embeddings": []}

    for spk_label, emb, chunk_id in all_embeddings:
        matched = False
        for cluster in clusters:
            center = _compute_center(cluster["embeddings"])
            sim = F.cosine_similarity(emb, center, dim=1).item()
            if sim > threshold:
                cluster["members"].append({"speaker_label": spk_label, "chunk_id": chunk_id})
                cluster["embeddings"].append(emb)
                matched = True
                break
        if not matched:
            cluster_label = f"GLOBAL_{len(clusters):02d}"
            clusters.append({
                "label": cluster_label,
                "members": [{"speaker_label": spk_label, "chunk_id": chunk_id}],
                "embeddings": [emb],
            })

    result = {c["label"]: c for c in clusters}
    logger.info("Clustered %d speakers into %d global identities",
                len(all_embeddings), len(result))
    return result


def _compute_center(embeddings):
    """计算一组 embedding 的均值向量。"""
    stacked = torch.cat(embeddings, dim=0)
    center = stacked.mean(dim=0, keepdim=True)
    return F.normalize(center)
