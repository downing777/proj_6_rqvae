#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate DPO (chosen vs rejected) title pairs for SID-conditioned training.

本地 vLLM 默认直连（无需 export 环境变量；也可被 --openai-base-url 覆盖）:
  DEFAULT_OPENAI_BASE_URL = http://localhost:8000/v1
  DEFAULT_OPENAI_API_KEY  = EMPTY

Token 用量记录位置（每成功一次 API 调用追加一行 JSON，且立即 flush）:
  - 默认: 与 --output-jsonl 同路径、去掉 .jsonl 后加 .usage.jsonl
    例: --output-jsonl .../dpo_electronics_generated.jsonl
        -> .../dpo_electronics_generated.usage.jsonl
  - 若指定 --usage-jsonl PATH，则写入 PATH

进度条: 使用 tqdm(默认) 每完成一个任务(含已跳过)更新一次; nohup 时可用 `tail -f` 看日志, 或加 --no-progress 只打普通日志。

nohup 后台运行示例(日志含进度):
  nohup python3 dpo_title_gen.py [参数...] > dpo_title_gen.nohup.log 2>&1 & echo $!
  # 查看: tail -f dpo_title_gen.nohup.log
  # 不显示进度条(日志更「干净」):
  # nohup python3 dpo_title_gen.py --no-progress [参数...] > dpo_title_gen.nohup.log 2>&1 &

生成长度: 仅由 --max-tokens 指定(原样作为 API 的 max_tokens)，请自行与模型/vLLM 上下文搭配。

Requires: pip install openai tqdm
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

try:
    from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI
except ImportError as e:  # pragma: no cover
    print("Error: openai is required. Install with: pip install openai", file=sys.stderr)
    raise e

try:
    from tqdm import tqdm
except ImportError as e:  # pragma: no cover
    print("Error: tqdm is required. Install with: pip install tqdm", file=sys.stderr)
    raise e


# 直接写在脚本中的默认 API 端点（无需 export OPENAI_*）
DEFAULT_OPENAI_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OPENAI_API_KEY = "EMPTY"

SidTuple = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #


def load_user_sid_map(path: str, id_column: str) -> Dict[str, SidTuple]:
    out: Dict[str, SidTuple] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        if id_column not in reader.fieldnames:
            raise ValueError(
                f"Column {id_column!r} not in CSV. Available: {reader.fieldnames}"
            )
        for need in ("rqid_0", "rqid_1", "rqid_2"):
            if need not in reader.fieldnames:
                raise ValueError(f"CSV must contain {need}. Got: {reader.fieldnames}")
        for row in reader:
            raw_id = (row.get(id_column) or "").strip()
            if not raw_id:
                continue
            try:
                sid = (int(row["rqid_0"]), int(row["rqid_1"]), int(row["rqid_2"]))
            except (KeyError, ValueError) as ex:
                raise ValueError(f"Bad SID row: {row}") from ex
            out[raw_id] = sid
    return out


def load_user_sid_map_jsonl(path: str) -> Dict[str, SidTuple]:
    out: Dict[str, SidTuple] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = (rec.get("user_id_raw") or "").strip()
            if not uid:
                continue
            rid = rec.get("rqvae_id")
            if isinstance(rid, list) and len(rid) == 3:
                try:
                    sid = (int(rid[0]), int(rid[1]), int(rid[2]))
                except (TypeError, ValueError):
                    continue
            elif all(f"rqid_{i}" in rec for i in range(3)):
                try:
                    sid = (int(rec["rqid_0"]), int(rec["rqid_1"]), int(rec["rqid_2"]))
                except (TypeError, ValueError, KeyError):
                    continue
            else:
                continue
            out[uid] = sid
    return out


def load_user_sid_map_auto(path: str, id_column: str) -> Dict[str, SidTuple]:
    pl = path.lower()
    if pl.endswith(".jsonl") or pl.endswith(".ndjson"):
        return load_user_sid_map_jsonl(path)
    if pl.endswith(".csv") or pl.endswith(".tsv"):
        return load_user_sid_map(path, id_column)
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    first = (first or "").lstrip()
    if first.startswith("{"):
        return load_user_sid_map_jsonl(path)
    return load_user_sid_map(path, id_column)


def iter_item_meta(
    path: str, max_items: Optional[int] = None
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parent = row.get("parent_asin")
            if not parent:
                continue
            yield str(parent), row
            n += 1
            if max_items is not None and n >= max_items:
                break


def stream_reviews_indexed(
    path: str,
    asin_set: Set[str],
    user_in_csv: Set[str],
) -> DefaultDict[str, DefaultDict[str, List[Dict[str, Any]]]]:
    by_asin: DefaultDict[str, DefaultDict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("parent_asin")
            if not asin or asin not in asin_set:
                continue
            uid = rec.get("user_id")
            if not uid or uid not in user_in_csv:
                continue
            text = (rec.get("text") or "").replace("<br />", " ").replace("<br/>", " ")
            text = re.sub(r"\s+", " ", text).strip()
            by_asin[asin][str(uid)].append(
                {
                    "rating": rec.get("rating"),
                    "text": text,
                    "title": (rec.get("title") or "")[:200],
                }
            )
    return by_asin


# --------------------------------------------------------------------------- #
#  Context building
# --------------------------------------------------------------------------- #


def _clip(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _features_bullets(item: Dict[str, Any], max_bullets: int = 3) -> str:
    feats = item.get("features") or []
    if not isinstance(feats, list):
        return ""
    out: List[str] = []
    for f in feats[:max_bullets]:
        if isinstance(f, str) and f.strip():
            out.append(_clip(f, 200))
    return " | ".join(out)


def build_product_context(
    item: Dict[str, Any],
    review_evidence: str,
    sid: SidTuple,
    other_evidence: Optional[Tuple[SidTuple, str]] = None,
) -> str:
    main_cat = item.get("main_category") or ""
    store = item.get("store") or ""
    otitle = (item.get("title") or "")[:300]
    price = item.get("price")
    price_s = f"{price}" if price is not None else "未知"
    cats = item.get("categories")
    if isinstance(cats, list) and cats:
        cat_s = " > ".join(str(c) for c in cats[:4])
    else:
        cat_s = str(main_cat)
    parts = [
        f"站点: 美国站; 主类: {main_cat}",
        f"父 ASIN: {item.get('parent_asin', '')}; 品牌/店铺: {store}",
        f"原商品标题(英文): {otitle}",
        f"价格: {price_s}; 浏览类目: {cat_s}",
        f"商品要点(英文摘录): {_features_bullets(item)}",
        f"目标 SID (rqid_0, rqid_1, rqid_2): {list(sid)}",
        f"该 SID 用户群在本商品上的评论证据(可有多条, 已截断): {review_evidence}",
    ]
    if other_evidence is not None:
        osid, otxt = other_evidence
        parts.append(
            f"同商品另一 SID 人群 {list(osid)} 的评论证据(用于构造 hard 负样): {otxt}"
        )
    return "\n".join(parts)


def aggregate_reviews(
    by_user: Dict[str, List[Dict[str, Any]]],
    user_ids: Set[str],
    max_chars: int,
) -> str:
    chunks: List[str] = []
    for uid in user_ids:
        for r in by_user.get(uid, []):
            t = (r.get("text") or "").strip()
            if t:
                rt = r.get("rating")
                prefix = f"[评分{rt}] " if isinstance(rt, (int, float)) else ""
                chunks.append(f"{prefix}{t}")

    def score(c: str) -> Tuple[float, int]:
        m = re.match(r"^\[评分([0-9.]+)\]", c)
        r = float(m.group(1)) if m else 0.0
        return (-r, -len(c))

    chunks.sort(key=score)
    out = " ".join(chunks)
    if len(out) <= max_chars:
        return out
    return _clip(out, max_chars)


def build_sid_to_users(
    asin_reviews: Dict[str, List[Dict[str, Any]]],
    user_to_sid: Dict[str, SidTuple],
) -> Dict[SidTuple, Set[str]]:
    m: DefaultDict[SidTuple, Set[str]] = defaultdict(set)
    for uid in asin_reviews:
        st = user_to_sid.get(uid)
        if st is not None and asin_reviews[uid]:
            m[st].add(uid)
    return dict(m)


# --------------------------------------------------------------------------- #
#  Tasks
# --------------------------------------------------------------------------- #


@dataclass
class GenTask:
    item_id: str
    item: Dict[str, Any]
    sid: SidTuple
    context: str
    has_cross: bool
    other_sid: Optional[SidTuple]


def pick_other_sid(
    sids: List[SidTuple],
    current: SidTuple,
    seed: int,
    asin: str,
) -> Optional[SidTuple]:
    others = [s for s in sids if s != current]
    if not others:
        return None
    h = int(hashlib.md5(f"{asin}:{current}".encode()).hexdigest()[:8], 16)
    rng = random.Random((h ^ seed) & 0xFFFFFFFF)
    return rng.choice(others)


def build_tasks(
    items_by_asin: Dict[str, Dict[str, Any]],
    reviews_by_asin: Dict[str, Any],
    user_to_sid: Dict[str, SidTuple],
    max_review_chars: int,
    skip_hard_if_single_sid: bool,
    seed: int,
    min_chars_evidence: int,
) -> List[GenTask]:
    tasks: List[GenTask] = []
    for asin, item in items_by_asin.items():
        urev = reviews_by_asin.get(asin)
        if not urev:
            continue
        sid_to_users = build_sid_to_users(urev, user_to_sid)
        sids = sorted(sid_to_users.keys(), key=lambda t: t)
        if not sids:
            continue
        for sid in sids:
            u_set = sid_to_users.get(sid, set())
            ev = aggregate_reviews(urev, u_set, max_review_chars)
            if len(ev.strip()) < min_chars_evidence:
                continue
            can_hard = len(sids) > 1
            if skip_hard_if_single_sid and len(sids) == 1:
                can_hard = False
            other: Optional[SidTuple] = None
            otxt = ""
            if can_hard:
                other = pick_other_sid(sids, sid, seed, asin)
                if other is not None and other in sid_to_users:
                    ou = sid_to_users[other]
                    otxt = aggregate_reviews(urev, ou, min(max_review_chars, 1500))
            has_cross = bool(can_hard and other is not None and otxt)
            ctx = build_product_context(
                item,
                ev,
                sid,
                (other, otxt) if has_cross else None,
            )
            tasks.append(
                GenTask(
                    item_id=asin,
                    item=item,
                    sid=sid,
                    context=ctx,
                    has_cross=has_cross,
                    other_sid=other if has_cross else None,
                )
            )
    return tasks


# --------------------------------------------------------------------------- #
#  LLM
# --------------------------------------------------------------------------- #


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


SYSTEM_PROMPT = (
    "你是美国亚马逊电商的标题与营销文案设计助手。根据给定的「商品信息」和「人群 SID」及评论证据，"
    "为「目标 SID」用户群写更有吸引力的中文商品短标题(用于主图旁或推荐位)。\n"
    "要求：\n"
    "1) 标题为中文，长度约 20~50 个汉字(可适当含英文品牌名/型号，勿编造不存在的功能)。\n"
    "2) title_chosen 必须体现目标 SID 人群在评论里关心的点。\n"
    "3) title_rejected_easy: 与商品弱相关、套路化、吸引力差的标题(如泛泛的「新品热销」式)。\n"
    "4) title_rejected_hard: 若上下文中给了「另一 SID」的评论，则写更像是在讨好那一类人群、而非目标 SID 的标题; "
    "若 JSON 中要求填 null(无 hard 时)，则输出 null。\n"
    "只输出一个 JSON 对象，键: title_chosen, title_rejected_easy, title_rejected_hard (可为 null)。不要其它文字。"
)

USER_WRAPPER = (
    "【商品与人群信息】\n{context}\n\n"
    "请只输出 JSON: "
    "{{\"title_chosen\": \"...\", \"title_rejected_easy\": \"...\", \"title_rejected_hard\": {hard_literal} }}"
)


def extract_usage(resp: Any) -> Optional[Dict[str, int]]:
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    out: Dict[str, int] = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(u, k, None)
        if v is not None:
            out[k] = int(v)
    return out if out else None


async def call_model_once(
    client: AsyncOpenAI,
    model: str,
    task: GenTask,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
    hard_literal = "null" if not task.has_cross else '"..."'
    user_msg = USER_WRAPPER.format(context=task.context, hard_literal=hard_literal)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "timeout": request_timeout,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = await client.chat.completions.create(**kwargs)
    ch = (resp.choices[0].message.content or "").strip()
    if not ch:
        raise ValueError("Empty model response")
    return parse_json_object(ch), extract_usage(resp)


async def run_with_retry(
    client: AsyncOpenAI,
    model: str,
    task: GenTask,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    max_retries: int,
    base_delay: float,
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
    last: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await call_model_once(
                client,
                model,
                task,
                max_tokens,
                temperature,
                top_p,
                request_timeout,
                extra_body,
            )
        except (APIError, APIConnectionError, APITimeoutError, json.JSONDecodeError, ValueError) as e:
            last = e
            if attempt >= max_retries:
                break
            delay = base_delay * (2**attempt) + random.uniform(0, 0.3 * base_delay)
            await asyncio.sleep(delay)
    assert last is not None
    raise last


def default_usage_jsonl_path(output_jsonl: str) -> str:
    if output_jsonl.endswith(".jsonl"):
        return output_jsonl[: -len(".jsonl")] + ".usage.jsonl"
    return output_jsonl + ".usage.jsonl"


def task_key(item_id: str, sid: SidTuple, neg_type: str) -> str:
    return f"{item_id}::{sid[0]},{sid[1]},{sid[2]}::{neg_type}"


def load_existing_keys(path: str) -> Set[str]:
    if not os.path.isfile(path):
        return set()
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("item_id")
            sid = row.get("sid")
            neg = row.get("negative_type")
            if not iid or not isinstance(sid, list) or len(sid) != 3 or not neg:
                continue
            out.add(task_key(str(iid), (int(sid[0]), int(sid[1]), int(sid[2])), str(neg)))
    return out


def dpo_rows_from_parsed(
    item_id: str,
    sid: SidTuple,
    context: str,
    has_cross: bool,
    other_sid: Optional[SidTuple],
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    chosen = (data.get("title_chosen") or "").strip()
    rej_e = (data.get("title_rejected_easy") or "").strip()
    rh = data.get("title_rejected_hard")
    rej_h = (rh or "").strip() if isinstance(rh, str) else ""
    if not chosen or not rej_e:
        raise ValueError("Missing title_chosen or title_rejected_easy")
    rows: List[Dict[str, Any]] = [
        {
            "item_id": item_id,
            "sid": list(sid),
            "context": context,
            "title_chosen": chosen,
            "title_rejected": rej_e,
            "negative_type": "easy_random",
            "meta": {"source": "dpo_title_gen"},
        }
    ]
    if has_cross and rej_h:
        rows.append(
            {
                "item_id": item_id,
                "sid": list(sid),
                "context": context,
                "title_chosen": chosen,
                "title_rejected": rej_h,
                "negative_type": "hard_cross_sid",
                "meta": {
                    "source": "dpo_title_gen",
                    "cross_against": list(other_sid) if other_sid else None,
                },
            }
        )
    return rows


async def amain() -> int:
    default_user_sid = (
        "/nfs5/yhy/tn/proj_6_rqvae/amazon_user/sid/user_semantic_ids.jsonl"
    )
    default_item = (
        "/nfs5/yhy/tn/proj_6_rqvae/amazon_user/raw/step4/final_filtered_item_meta_electronics.jsonl"
    )
    default_rev = (
        "/nfs5/yhy/tn/proj_6_rqvae/amazon_user/raw/step4/final_target_user_reviews_by_category/"
        "final_target_user_reviews_electronics.jsonl"
    )
    out_default = "/nfs5/yhy/tn/proj_6_rqvae/softprompt/data/dpo_electronics_generated.jsonl"
    ap = argparse.ArgumentParser(
        description="Generate DPO title pairs (Chinese) for SID groups via OpenAI-compatible API."
    )
    ap.add_argument(
        "--user-sid",
        "--user-sid-csv",
        dest="user_sid",
        type=str,
        default=default_user_sid,
        help="User -> SID: .jsonl (user_id_raw + rqvae_id) or .csv (rqid_0/1/2 + id column).",
    )
    ap.add_argument(
        "--id-column",
        type=str,
        default="user_id_raw",
        help="CSV: column to join to reviews user_id.",
    )
    ap.add_argument("--item-jsonl", type=str, default=default_item)
    ap.add_argument("--reviews-jsonl", type=str, default=default_rev)
    ap.add_argument("--output-jsonl", type=str, default=out_default)
    ap.add_argument(
        "--openai-base-url",
        type=str,
        default=DEFAULT_OPENAI_BASE_URL,
        help=f"Default in script: {DEFAULT_OPENAI_BASE_URL}",
    )
    ap.add_argument(
        "--openai-api-key",
        type=str,
        default=DEFAULT_OPENAI_API_KEY,
        help="Default in script: EMPTY (vLLM).",
    )
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--retry-base-delay", type=float, default=1.0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--model", type=str, default="Qwen/Qwen25-32B")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="传给 API 的 max_tokens，即生成长度上限，由你自行设定。",
    )
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-review-chars", type=int, default=3000)
    ap.add_argument("--min-review-chars", type=int, default=8)
    ap.add_argument(
        "--skip-hard-if-single-sid", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument(
        "--extra-body-json",
        type=str,
        default="",
        help='vLLM extra_body; 默认 {"top_k": 20} 若本参数为空',
    )
    ap.add_argument(
        "--usage-jsonl",
        type=str,
        default="",
        help="Token 日志; 空则: <output-jsonl> 同目录下 *.usage.jsonl",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示进度条(适合完全重定向到纯文本日志、避免非 TTY 下 tqdm 行为干扰)。",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.user_sid):
        print(f"Error: --user-sid not found: {args.user_sid}", file=sys.stderr)
        return 1

    user_to_sid = load_user_sid_map_auto(args.user_sid, args.id_column)
    if not user_to_sid:
        print("Error: no user -> SID entries loaded.", file=sys.stderr)
        return 1
    print(f"Loaded {len(user_to_sid)} user -> SID mappings from {args.user_sid}.")

    items_by_asin: Dict[str, Dict[str, Any]] = {}
    for asin, row in iter_item_meta(args.item_jsonl, max_items=args.max_items):
        items_by_asin[asin] = row
    asin_set = set(items_by_asin)
    print(f"Loaded {len(asin_set)} items from {args.item_jsonl}.")

    print("Streaming reviews (one pass)…")
    t0 = time.time()
    reviews_by_asin = stream_reviews_indexed(
        args.reviews_jsonl, asin_set, set(user_to_sid)
    )
    print(f"Indexed reviews for {len(reviews_by_asin)} ASINs in {time.time() - t0:.1f}s.")

    if args.extra_body_json.strip():
        extra_body: Optional[Dict[str, Any]] = json.loads(args.extra_body_json)
    else:
        extra_body = {"top_k": 20}

    client = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
    )
    print(f"OpenAI base_url: {args.openai_base_url}")

    tasks = build_tasks(
        items_by_asin,
        reviews_by_asin,
        user_to_sid,
        args.max_review_chars,
        args.skip_hard_if_single_sid,
        args.seed,
        args.min_review_chars,
    )
    print(f"Built {len(tasks)} generation tasks (item, SID).")

    existing: Set[str] = set()
    if args.skip_existing and os.path.isfile(args.output_jsonl):
        existing = load_existing_keys(args.output_jsonl)
        print(f"Skip-existing: {len(existing)} keys in {args.output_jsonl}.")

    usage_path = (args.usage_jsonl or default_usage_jsonl_path(args.output_jsonl)).strip()
    print(f"Token 用量将写入: {usage_path} (每行一条 API 的 usage，含 prompt/completion/total)")

    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl)) or "."
    os.makedirs(out_dir, exist_ok=True)
    u_dir = os.path.dirname(os.path.abspath(usage_path)) or "."
    if u_dir and u_dir != out_dir:
        os.makedirs(u_dir, exist_ok=True)

    file_lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, args.max_concurrency))
    _sl = asyncio.Lock()
    _st: Dict[str, Any] = {
        "ok": 0,
        "err": 0,
        "skip": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests_with_usage": 0,
    }

    async def process_one(t: GenTask) -> None:
        k_e = task_key(t.item_id, t.sid, "easy_random")
        k_h = task_key(t.item_id, t.sid, "hard_cross_sid")
        need_e = (not args.skip_existing) or (k_e not in existing)
        need_h = t.has_cross and ((not args.skip_existing) or (k_h not in existing))
        if not need_e and not need_h:
            async with _sl:
                _st["skip"] += 1
            return
        async with sem:
            try:
                data, usage = await run_with_retry(
                    client=client,
                    model=args.model,
                    task=t,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    request_timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    base_delay=args.retry_base_delay,
                    extra_body=extra_body,
                )
                rows = dpo_rows_from_parsed(
                    t.item_id, t.sid, t.context, t.has_cross, t.other_sid, data
                )
                to_write: List[Dict[str, Any]] = []
                for row in rows:
                    neg = str(row.get("negative_type", ""))
                    k = task_key(t.item_id, t.sid, neg)
                    if args.skip_existing and k in existing:
                        continue
                    to_write.append(row)
                if not to_write:
                    async with _sl:
                        _st["skip"] += 1
                    return
                async with file_lock:
                    with open(args.output_jsonl, "a", encoding="utf-8") as f:
                        for row in to_write:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            f.flush()
                            existing.add(
                                task_key(
                                    t.item_id, t.sid, str(row.get("negative_type", ""))
                                )
                            )
                    rec: Dict[str, Any] = {
                        "ts": time.time(),
                        "item_id": t.item_id,
                        "sid": list(t.sid),
                        "model": args.model,
                        "dpo_rows_written": len(to_write),
                    }
                    if usage:
                        rec["usage"] = usage
                    else:
                        rec["usage"] = None
                        rec["usage_note"] = "server_returned_no_usage; check vLLM version / flags"
                    with open(usage_path, "a", encoding="utf-8") as uf:
                        uf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        uf.flush()
                async with _sl:
                    _st["ok"] += 1
                    if usage:
                        _st["requests_with_usage"] += 1
                        _st["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                        _st["completion_tokens"] += int(
                            usage.get("completion_tokens", 0) or 0
                        )
                        _st["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            except Exception:
                print(
                    f"[ERROR] item={t.item_id} sid={t.sid}:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                async with _sl:
                    _st["err"] += 1

    async def run_with_progress() -> None:
        if args.no_progress:
            await asyncio.gather(*[process_one(t) for t in tasks])
            return
        pbar = tqdm(
            total=len(tasks),
            desc="DPO 生成",
            unit="条",
            file=sys.stdout,
            mininterval=0.3,
        )

        async def run_one(t: GenTask) -> None:
            try:
                await process_one(t)
            finally:
                pbar.update(1)
                pbar.set_postfix(
                    ok=_st["ok"],
                    err=_st["err"],
                    skip=_st["skip"],
                    refresh=False,
                )

        try:
            await asyncio.gather(*[run_one(t) for t in tasks])
        finally:
            pbar.close()

    await run_with_progress()
    print(
        json.dumps(
            {
                "output_jsonl": args.output_jsonl,
                "usage_jsonl": usage_path,
                "token_usage_aggregated": {
                    "requests_with_usage": _st["requests_with_usage"],
                    "sum_prompt_tokens": _st["prompt_tokens"],
                    "sum_completion_tokens": _st["completion_tokens"],
                    "sum_total_tokens": _st["total_tokens"],
                },
                "successful_api_calls": _st["ok"],
                "errors": _st["err"],
                "skipped_tasks": _st["skip"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if _st["err"] == 0 else 2


def main() -> None:
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        print("Interrupted. 已 flush 的 DPO/usage 行已保留; 可加 --skip-existing 续跑。", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
