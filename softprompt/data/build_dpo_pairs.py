import argparse
import json
import os
from typing import Dict, List


SFT_SCHEMA: Dict[str, object] = {
    "type": "object",
    "required": ["item_id", "sid", "context", "target_title"],
    "properties": {
        "item_id": {"type": "string"},
        "sid": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
        },
        "context": {"type": "string"},
        "target_title": {"type": "string"},
        "meta": {"type": "object"},
    },
}

DPO_SCHEMA: Dict[str, object] = {
    "type": "object",
    "required": ["item_id", "sid", "context", "title_chosen", "title_rejected"],
    "properties": {
        "item_id": {"type": "string"},
        "sid": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
        },
        "context": {"type": "string"},
        "title_chosen": {"type": "string"},
        "title_rejected": {"type": "string"},
        "negative_type": {"type": "string", "enum": ["hard_cross_sid", "easy_random"]},
        "meta": {"type": "object"},
    },
}


def build_sft_examples() -> List[Dict[str, object]]:
    return [
        {
            "item_id": "B0-123",
            "sid": [1, 2, 3],
            "context": "品牌: XX; 品类: 蓝牙耳机; 卖点: 长续航, 降噪, 舒适佩戴",
            "target_title": "长续航降噪蓝牙耳机，通勤运动都舒适",
            "meta": {"split": "train"},
        },
        {
            "item_id": "B0-456",
            "sid": [8, 6, 4],
            "context": "品牌: YY; 品类: 咖啡机; 卖点: 一键萃取, 小巧, 易清洗",
            "target_title": "一键萃取小巧咖啡机，在家轻松喝好咖啡",
            "meta": {"split": "train"},
        },
    ]


def build_dpo_examples() -> List[Dict[str, object]]:
    return [
        {
            "item_id": "B0-123",
            "sid": [1, 2, 3],
            "context": "品牌: XX; 品类: 蓝牙耳机; 卖点: 长续航, 降噪, 舒适佩戴",
            "title_chosen": "长续航降噪蓝牙耳机，通勤运动都舒适",
            "title_rejected": "耳机新品上市，喜欢就来看看",
            "negative_type": "easy_random",
            "meta": {"source": "ctr_pair"},
        },
        {
            "item_id": "B0-123",
            "sid": [1, 2, 3],
            "context": "品牌: XX; 品类: 蓝牙耳机; 卖点: 长续航, 降噪, 舒适佩戴",
            "title_chosen": "长续航降噪蓝牙耳机，通勤运动都舒适",
            "title_rejected": "高颜值入耳式耳机，拍照穿搭更吸睛",
            "negative_type": "hard_cross_sid",
            "meta": {"source": "cross_sid_negative"},
        },
    ]


def dump_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def dump_jsonl(path: str, payloads: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in payloads:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_readme(path: str) -> None:
    content = """# SID 标题训练数据格式

本目录只定义格式，不做真实样本构建。

## 文件
- sft_samples.schema.json: SFT 数据 schema
- dpo_pairs.schema.json: DPO 数据 schema
- sft_samples.example.jsonl: SFT 示例数据
- dpo_pairs.example.jsonl: DPO 示例数据

## 约束建议
- sid 使用固定三元组: [rqid_0, rqid_1, rqid_2]
- DPO 负样本建议混合:
  - hard_cross_sid: 70%
  - easy_random: 30%
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write SID title task schemas and examples.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="softprompt/data/format",
        help="Output folder for schema/example files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dump_json(os.path.join(args.output_dir, "sft_samples.schema.json"), SFT_SCHEMA)
    dump_json(os.path.join(args.output_dir, "dpo_pairs.schema.json"), DPO_SCHEMA)
    dump_jsonl(os.path.join(args.output_dir, "sft_samples.example.jsonl"), build_sft_examples())
    dump_jsonl(os.path.join(args.output_dir, "dpo_pairs.example.jsonl"), build_dpo_examples())
    dump_readme(os.path.join(args.output_dir, "README.md"))
    print(f"Saved format files to: {args.output_dir}")


if __name__ == "__main__":
    main()
