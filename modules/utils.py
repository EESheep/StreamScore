"""全局工具函数。无外部项目依赖。"""
import json
import logging
import os
import yaml


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(file_path):
    """逐行读取 JSONL 文件，返回 list[dict]."""
    records = []
    if not os.path.exists(file_path):
        return records
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(file_path, records):
    """将 list[dict] 写入 JSONL 文件。"""
    with open(file_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(file_path, record):
    """追加单条记录到 JSONL 文件。"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
