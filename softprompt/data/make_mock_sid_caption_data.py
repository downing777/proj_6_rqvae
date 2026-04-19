import argparse
import json
import os
import random
from typing import Dict, List, Sequence, Tuple


US_ITEMS: Sequence[Dict[str, str]] = [
    {
        "item_id": "US-ELEC-001",
        "brand": "SoundPeak",
        "category": "蓝牙耳机",
        "features": "主动降噪, 36小时续航, 通话清晰",
    },
    {
        "item_id": "US-HOME-002",
        "brand": "BrewCraft",
        "category": "滴滤咖啡机",
        "features": "一键冲煮, 保温40分钟, 可拆洗滤网",
    },
    {
        "item_id": "US-KITCH-003",
        "brand": "FreshMix",
        "category": "空气炸锅",
        "features": "可视窗口, 低油烹饪, 8种预设菜单",
    },
    {
        "item_id": "US-SPORT-004",
        "brand": "FitMotion",
        "category": "智能手环",
        "features": "心率监测, 睡眠分析, 50米防水",
    },
    {
        "item_id": "US-CARE-005",
        "brand": "PureGlow",
        "category": "电动牙刷",
        "features": "三档清洁, 2分钟计时, 续航30天",
    },
    {
        "item_id": "US-OFFICE-006",
        "brand": "KeyNova",
        "category": "机械键盘",
        "features": "热插拔轴体, RGB背光, 静音卫星轴",
    },
]


SID_STYLES: Sequence[Dict[str, str]] = [
    {"tone": "极致性价比", "hook": "预算友好", "benefit": "花更少的钱拿下核心功能"},
    {"tone": "效率优先", "hook": "节省时间", "benefit": "一步到位，减少重复操作"},
    {"tone": "品质稳定", "hook": "耐用安心", "benefit": "长期使用更省心"},
    {"tone": "颜值导向", "hook": "外观好看", "benefit": "摆在家里也很有质感"},
    {"tone": "轻量便捷", "hook": "使用门槛低", "benefit": "新手也能快速上手"},
    {"tone": "功能党", "hook": "参数够硬", "benefit": "关键功能覆盖更完整"},
]


def parse_sid_dims(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"--sid-dims must be three ints, got: {text}")
    return parts[0], parts[1], parts[2]


def sid_to_style_index(sid: Sequence[int]) -> int:
    return int(sum((i + 1) * sid[i] for i in range(len(sid)))) % len(SID_STYLES)


def build_context(item: Dict[str, str], sid: Sequence[int]) -> str:
    style = SID_STYLES[sid_to_style_index(sid)]
    return (
        f"站点: 美国; 品牌: {item['brand']}; 品类: {item['category']}; "
        f"核心卖点: {item['features']}; 人群偏好: {style['tone']}, {style['hook']}"
    )


def build_title(item: Dict[str, str], sid: Sequence[int]) -> str:
    style = SID_STYLES[sid_to_style_index(sid)]
    feature_head = item["features"].split(",")[0].strip()
    return (
        f"{item['brand']}{item['category']}，{feature_head}，{style['hook']}更突出，"
        f"{style['benefit']}"
    )


def build_easy_negative(item: Dict[str, str]) -> str:
    return f"{item['category']}新品上架，欢迎选购"


def build_hard_negative(item: Dict[str, str], sid: Sequence[int], rng: random.Random) -> str:
    current = sid_to_style_index(sid)
    alt_candidates = [i for i in range(len(SID_STYLES)) if i != current]
    alt = SID_STYLES[rng.choice(alt_candidates)]
    feature_head = item["features"].split(",")[0].strip()
    return (
        f"{item['brand']}{item['category']}，{feature_head}，{alt['hook']}更突出，"
        f"{alt['benefit']}"
    )


def random_sid(rng: random.Random, sid_dims: Tuple[int, int, int]) -> List[int]:
    return [rng.randrange(0, sid_dims[0]), rng.randrange(0, sid_dims[1]), rng.randrange(0, sid_dims[2])]


def write_jsonl(path: str, rows: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock SID-caption data for SFT/DPO smoke runs.")
    parser.add_argument("--output-dir", type=str, default="softprompt/data/mock")
    parser.add_argument("--sid-dims", type=str, default="32,32,32")
    parser.add_argument("--num-items", type=int, default=6)
    parser.add_argument("--sids-per-item", type=int, default=4)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.hard_negative_ratio <= 1.0:
        raise ValueError("--hard-negative-ratio must be in [0, 1].")
    sid_dims = parse_sid_dims(args.sid_dims)
    if args.num_items <= 0 or args.sids_per_item <= 0:
        raise ValueError("--num-items and --sids-per-item must be > 0.")

    rng = random.Random(args.seed)
    items = list(US_ITEMS[: min(args.num_items, len(US_ITEMS))])
    if args.num_items > len(items):
        for i in range(len(items), args.num_items):
            base = US_ITEMS[i % len(US_ITEMS)]
            clone = dict(base)
            clone["item_id"] = f"{base['item_id']}-{i+1}"
            items.append(clone)

    sft_rows: List[Dict[str, object]] = []
    dpo_rows: List[Dict[str, object]] = []

    for item in items:
        seen_sids = set()
        while len(seen_sids) < args.sids_per_item:
            sid = tuple(random_sid(rng, sid_dims))
            seen_sids.add(sid)

        for sid_t in seen_sids:
            sid = list(sid_t)
            context = build_context(item=item, sid=sid)
            chosen = build_title(item=item, sid=sid)
            sft_rows.append(
                {
                    "item_id": item["item_id"],
                    "sid": sid,
                    "context": context,
                    "target_title": chosen,
                    "meta": {"split": "train", "mock": True},
                }
            )

            use_hard = rng.random() < args.hard_negative_ratio
            negative_type = "hard_cross_sid" if use_hard else "easy_random"
            if use_hard:
                rejected = build_hard_negative(item=item, sid=sid, rng=rng)
            else:
                rejected = build_easy_negative(item=item)

            dpo_rows.append(
                {
                    "item_id": item["item_id"],
                    "sid": sid,
                    "context": context,
                    "title_chosen": chosen,
                    "title_rejected": rejected,
                    "negative_type": negative_type,
                    "meta": {"mock": True, "seed": args.seed},
                }
            )

    os.makedirs(args.output_dir, exist_ok=True)
    sft_path = os.path.join(args.output_dir, "sft_train.jsonl")
    dpo_path = os.path.join(args.output_dir, "dpo_train.jsonl")
    write_jsonl(sft_path, sft_rows)
    write_jsonl(dpo_path, dpo_rows)
    print(
        json.dumps(
            {
                "sft_path": sft_path,
                "dpo_path": dpo_path,
                "sft_count": len(sft_rows),
                "dpo_count": len(dpo_rows),
                "hard_negative_ratio": args.hard_negative_ratio,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
