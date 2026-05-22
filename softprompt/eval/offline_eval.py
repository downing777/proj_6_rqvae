#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-judge offline evaluation for SID-conditioned title generation.

输入:
  --pred-jsonl: softprompt/infer/generate_title.py 的产物, 每行含
      {item_id, sid, context, generated_text, original_title}

  --reviews-jsonl + --user-sid: Amazon 评论 + user_id_raw -> SID 映射,
    给 judge 注入 "该 SID 用户群在本商品下的真实评论" 作为用户画像证据
    (口径与 dpo_title_gen.py 完全一致, 复用其 stream_reviews_indexed / aggregate_reviews)

评判维度: personalization / hallucination / fluency / overall。
为消除位置偏置, 每条样本随机决定 original/generated 落 A 或 B, 再把 judge 回答映射回 (original, generated)。

依赖: pip install openai tqdm
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

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

from softprompt.dpo_title_gen import aggregate_reviews, stream_reviews_indexed


DEFAULT_OPENAI_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OPENAI_API_KEY = "EMPTY"

SidTuple = Tuple[int, ...]


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def to_sid_tuple(sid: Any) -> Optional[SidTuple]:
    if not isinstance(sid, (list, tuple)) or not sid:
        return None
    try:
        return tuple(int(x) for x in sid)
    except (TypeError, ValueError):
        return None


_ORIGINAL_TITLE_RE = re.compile(r"^\s*Original title:\s*(.+)$", re.IGNORECASE)


def extract_original_title(context: str) -> str:
    """Fallback: 当 prediction 行没有 original_title 字段时, 从 context 里抠 'Original title: ...' 行。"""
    if not context:
        return ""
    for line in context.splitlines():
        m = _ORIGINAL_TITLE_RE.match(line)
        if m:
            return m.group(1).strip().strip(".")
    return ""


def context_without_original_title(context: str) -> str:
    """剥掉 context 里的 'Original title:' 行再给 judge —— 否则 judge 一眼看到原文,
    候选 A/B 的位置随机化形同虚设。"""
    if not context:
        return ""
    kept = [line for line in context.splitlines() if not _ORIGINAL_TITLE_RE.match(line)]
    return "\n".join(kept).strip()


# --------------------------------------------------------------------------- #
#  User-group evidence (real reviews for the (item, SID) tuple)
# --------------------------------------------------------------------------- #


def load_user_to_sid(user_sid_jsonl: str) -> Dict[str, SidTuple]:
    """user_semantic_ids.jsonl -> {user_id_raw: SidTuple}"""
    out: Dict[str, SidTuple] = {}
    for r in load_jsonl(user_sid_jsonl):
        raw = r.get("user_id_raw")
        rqv = r.get("rqvae_id")
        if not raw or not isinstance(rqv, list):
            continue
        try:
            out[str(raw)] = tuple(int(x) for x in rqv)
        except (TypeError, ValueError):
            continue
    return out


def format_review_evidence(
    reviews_by_asin: Any,
    user_to_sid: Dict[str, SidTuple],
    item_id: str,
    sid: SidTuple,
    max_chars: int,
) -> str:
    """对 (item_id, sid), 拉出映射到该 SID 的用户在该商品下的真实评论并聚合。
    与 dpo_title_gen 同源, 保证 judge 看到的用户画像跟造数据时老师 LLM 一致。"""
    urev = reviews_by_asin.get(item_id)
    if not urev:
        return ""
    u_set = {uid for uid in urev if user_to_sid.get(uid) == sid}
    if not u_set:
        return ""
    ev = aggregate_reviews(urev, u_set, max_chars).strip()
    if not ev:
        return ""
    return (
        "Real reviews from this SID user group on this product "
        "(sorted by rating, truncated):\n" + ev
    )


# --------------------------------------------------------------------------- #
#  Sample assembly
# --------------------------------------------------------------------------- #


@dataclass
class JudgeSample:
    item_id: str
    sid: SidTuple
    context: str
    original_title: str
    generated_title: str
    user_evidence: str


def build_samples(
    pred_rows: List[Dict[str, Any]],
    reviews_by_asin: Any,
    user_to_sid: Dict[str, SidTuple],
    evidence_max_chars: int,
) -> List[JudgeSample]:
    samples: List[JudgeSample] = []
    for r in pred_rows:
        sid = to_sid_tuple(r.get("sid"))
        iid = r.get("item_id")
        if not iid or sid is None:
            continue
        context = r.get("context") or ""
        # generate_title.py 已经只写 prompt 之后的新 token, 这里直接 strip 即可
        gen_title = (r.get("generated_text") or "").strip()
        if not gen_title:
            continue
        original = (r.get("original_title") or "").strip() or extract_original_title(context)
        if not original:
            continue
        user_ev = format_review_evidence(
            reviews_by_asin, user_to_sid, str(iid), sid, evidence_max_chars
        )
        samples.append(
            JudgeSample(
                item_id=str(iid),
                sid=sid,
                context=context,
                original_title=original,
                generated_title=gen_title,
                user_evidence=user_ev,
            )
        )
    return samples


# --------------------------------------------------------------------------- #
#  LLM judge
# --------------------------------------------------------------------------- #


JUDGE_SYSTEM_PROMPT = (
    "You are a strict, careful evaluator for e-commerce product titles. "
    "Given a product's objective info, real reviews from the target user group, "
    "and two candidate titles (labeled A and B), compare them on the THREE dimensions below. "
    "You do NOT need to give an overall preference — judge only the three dimensions independently.\n"
    "Dimensions:\n"
    "1) personalization: Which title better matches the interests / preferences of the target user group "
    "as revealed by their reviews? Values: A_better / B_better / tie. "
    "If user-group evidence is empty, output tie.\n"
    "2) hallucination: Does either title introduce specific attributes / features / materials / numbers "
    "(e.g. nonexistent battery life, fake certification, exaggerated specs) NOT supported by the product info? "
    "Values: A_better / B_better / tie (better = fewer hallucinations; if both are essentially clean, use tie).\n"
    "3) fluency: Is the English natural and well-formed (word order, collocations, no repeated tokens / gibberish)? "
    "Values: A_better / B_better / tie.\n"
    "Output STRICTLY a single JSON object with fields personalization, hallucination, fluency, reason. "
    "reason is a one-sentence explanation. No additional text."
)

JUDGE_USER_TEMPLATE = (
    "[Product objective info]\n{product_attrs}\n\n"
    "[Target SID user-group evidence]\n{user_evidence}\n\n"
    "[Candidate A]\n{title_a}\n\n"
    "[Candidate B]\n{title_b}\n\n"
    "Output JSON only: "
    "{{\"personalization\": \"A_better|B_better|tie\", "
    "\"hallucination\": \"A_better|B_better|tie\", "
    "\"fluency\": \"A_better|B_better|tie\", "
    "\"reason\": \"...\"}}"
)


def parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        text = obj_match.group(0)
    return json.loads(text)


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


def position_role(sample: JudgeSample, seed: int) -> Tuple[str, str, str]:
    """决定 original/generated 落 A 还是 B (位置随机化)。返回 (title_a, title_b, original_pos)。"""
    h = int(hashlib.md5(f"{sample.item_id}:{sample.sid}:{seed}".encode()).hexdigest()[:8], 16)
    if h % 2 == 0:
        return sample.original_title, sample.generated_title, "A"
    return sample.generated_title, sample.original_title, "B"


def _map_two(value: str, original_pos: str) -> str:
    """把 A_better / B_better / tie 翻译成 original_better / generated_better / tie。"""
    v = (value or "").strip().lower()
    if v not in ("a_better", "b_better", "tie"):
        return "tie"
    if v == "tie":
        return "tie"
    chose = "A" if v == "a_better" else "B"
    return "original_better" if chose == original_pos else "generated_better"


_DIMS = ("personalization", "hallucination", "fluency")


def _derive_overall(norm: Dict[str, str]) -> str:
    """从 3 维度判断推导 overall: 票数和决定方向, 幻觉一票否决。
    +1 generated_better / -1 original_better / 0 tie -> 求和。
    Hallucination veto: 如果 hallucination 上 original 更好, overall 不能给 generated_win
    (反向同理), 避免"幻觉更严重但拼了流畅/个性化反超"的可疑判定。"""
    score = 0
    for k in _DIMS:
        v = norm[k]
        if v == "generated_better":
            score += 1
        elif v == "original_better":
            score -= 1
    if norm["hallucination"] == "original_better" and score > 0:
        return "tie"
    if norm["hallucination"] == "generated_better" and score < 0:
        return "tie"
    if score > 0:
        return "generated_win"
    if score < 0:
        return "original_win"
    return "tie"


def normalize_judge(raw: Dict[str, Any], original_pos: str) -> Dict[str, Any]:
    norm: Dict[str, Any] = {
        k: _map_two(str(raw.get(k, "tie")), original_pos) for k in _DIMS
    }
    norm["reason"] = str(raw.get("reason", ""))[:500]
    norm["overall"] = _derive_overall(norm)
    return norm


def is_strict_win(norm: Dict[str, Any]) -> bool:
    """严格胜出: 3 个维度全部不输 (≥1 项更好, 其余至少 tie)。"""
    if any(norm[k] == "original_better" for k in _DIMS):
        return False
    return any(norm[k] == "generated_better" for k in _DIMS)


# --------------------------------------------------------------------------- #
#  Async pipeline
# --------------------------------------------------------------------------- #


async def call_judge_once(
    client: AsyncOpenAI,
    model: str,
    sample: JudgeSample,
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, int]]]:
    title_a, title_b, original_pos = position_role(sample, seed)
    user_msg = JUDGE_USER_TEMPLATE.format(
        product_attrs=context_without_original_title(sample.context) or "(none)",
        user_evidence=sample.user_evidence or "(none)",
        title_a=title_a,
        title_b=title_b,
    )
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
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
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise ValueError("Empty judge response")
    parsed = parse_json_object(raw)
    norm = normalize_judge(parsed, original_pos)
    meta = {"original_pos": original_pos, "raw_judge": parsed}
    return norm, meta, extract_usage(resp)


async def call_judge_with_retry(
    client: AsyncOpenAI,
    model: str,
    sample: JudgeSample,
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    max_retries: int,
    base_delay: float,
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, int]]]:
    last: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await call_judge_once(
                client, model, sample, seed,
                max_tokens, temperature, top_p,
                request_timeout, extra_body,
            )
        except (APIError, APIConnectionError, APITimeoutError, json.JSONDecodeError, ValueError) as e:
            last = e
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3 * base_delay)
            await asyncio.sleep(delay)
    assert last is not None
    raise last


def judge_key(item_id: str, sid: SidTuple) -> str:
    return f"{item_id}::{','.join(str(x) for x in sid)}"


def load_existing_judge_keys(path: str) -> Set[str]:
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
            sid = to_sid_tuple(row.get("sid"))
            if iid and sid:
                out.add(judge_key(str(iid), sid))
    return out


def default_usage_jsonl_path(output_jsonl: str) -> str:
    if output_jsonl.endswith(".jsonl"):
        return output_jsonl[: -len(".jsonl")] + ".usage.jsonl"
    return output_jsonl + ".usage.jsonl"


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """三维度 win rate + 推导出的 overall win rate, 都按 generated 视角统计。

    每个维度的 win_rate = generated_better / n。
    overall 由每条 judgment 的 _derive_overall (幻觉一票否决 + 票数和) 推出。
    """
    n = len(judgments)
    if n == 0:
        return {"sample_count": 0}

    overall_counts = Counter(j.get("overall", "tie") for j in judgments)
    overall_win = overall_counts.get("generated_win", 0)
    overall_tie = overall_counts.get("tie", 0)
    overall_lose = overall_counts.get("original_win", 0)
    decisive = overall_win + overall_lose
    win_rate_excl_tie = (overall_win / decisive) if decisive > 0 else None

    strict_wins = sum(1 for j in judgments if is_strict_win(j))
    sid_buckets = Counter(",".join(str(x) for x in j["sid"]) for j in judgments)

    per_dim: Dict[str, Dict[str, Any]] = {}
    for dim in _DIMS:
        gw = sum(1 for j in judgments if j.get(dim) == "generated_better")
        tw = sum(1 for j in judgments if j.get(dim) == "tie")
        ow = sum(1 for j in judgments if j.get(dim) == "original_better")
        per_dim[dim] = {
            "generated_win_rate": gw / n,
            "tie_rate": tw / n,
            "original_win_rate": ow / n,
            "wilson95_generated_win_rate": wilson_ci(gw, n),
        }

    return {
        "sample_count": n,
        # 三维度 + overall 的 generated win rate (主指标)
        "win_rates": {
            "personalization": per_dim["personalization"]["generated_win_rate"],
            "hallucination": per_dim["hallucination"]["generated_win_rate"],
            "fluency": per_dim["fluency"]["generated_win_rate"],
            "overall": overall_win / n,
        },
        # overall 的完整 break-down (tie / lose / 排除 tie 后的胜率 / 置信区间)
        "overall": {
            "generated_win_rate": overall_win / n,
            "tie_rate": overall_tie / n,
            "original_win_rate": overall_lose / n,
            "win_rate_excluding_tie": win_rate_excl_tie,
            "wilson95_generated_win_rate": wilson_ci(overall_win, n),
        },
        # 每维度完整 break-down
        "per_dimension": per_dim,
        # 严格胜出: 3 维度全部不输且至少一项更好
        "strict_win_rate": strict_wins / n,
        "wilson95_strict_win_rate": wilson_ci(strict_wins, n),
        "sid_bucket_count": len(sid_buckets),
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


async def amain() -> int:
    ap = argparse.ArgumentParser(description="LLM-as-judge offline eval for SID title generation.")
    ap.add_argument("--pred-jsonl", type=str, required=True,
                    help="generate_title.py 产出, 每行 {item_id, sid, context, generated_text, original_title}")
    ap.add_argument("--reviews-jsonl", type=str, required=True,
                    help="Amazon 评论 jsonl, 每行至少含 parent_asin/user_id/rating/text")
    ap.add_argument("--user-sid", type=str, required=True,
                    help="user_semantic_ids.jsonl (user_id_raw + rqvae_id)")
    ap.add_argument("--output-jsonl", type=str,
                    default="softprompt/outputs/llm_judge.jsonl",
                    help="逐条 judge 结果")
    ap.add_argument("--summary-json", type=str,
                    default="softprompt/outputs/llm_judge_summary.json",
                    help="聚合统计")
    ap.add_argument("--usage-jsonl", type=str, default="",
                    help="API token 用量, 默认 <output-jsonl>.usage.jsonl")
    ap.add_argument("--openai-base-url", type=str, default=DEFAULT_OPENAI_BASE_URL)
    ap.add_argument("--openai-api-key", type=str, default=DEFAULT_OPENAI_API_KEY)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                    help="judge 模型 (建议比被评模型更强)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--retry-base-delay", type=float, default=1.0)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="只评估前 N 条 (调试用)")
    ap.add_argument("--seed", type=int, default=42, help="位置随机化的种子")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--extra-body-json", type=str, default="",
                    help='vLLM extra_body; 为空则使用 {"top_k": 1} 以贴近贪心')
    ap.add_argument("--evidence-max-chars", type=int, default=1500,
                    help="评论证据聚合后的最大字符数 (跟 dpo_title_gen --max-review-chars 对齐)")
    args = ap.parse_args()

    for label, path in [
        ("--pred-jsonl", args.pred_jsonl),
        ("--reviews-jsonl", args.reviews_jsonl),
        ("--user-sid", args.user_sid),
    ]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    pred_rows = load_jsonl(args.pred_jsonl)
    print(f"Loaded {len(pred_rows)} predictions.")

    user_to_sid = load_user_to_sid(args.user_sid)
    print(f"Loaded user_to_sid: {len(user_to_sid)} users.")

    asin_set = {str(r.get("item_id")) for r in pred_rows if r.get("item_id")}
    print(f"Indexing reviews for {len(asin_set)} predicted items...")
    reviews_by_asin = stream_reviews_indexed(args.reviews_jsonl, asin_set, set(user_to_sid))
    n_with_reviews = sum(1 for asin in asin_set if reviews_by_asin.get(asin))
    print(f"  reviews indexed for {n_with_reviews}/{len(asin_set)} items.")

    samples = build_samples(
        pred_rows,
        reviews_by_asin=reviews_by_asin,
        user_to_sid=user_to_sid,
        evidence_max_chars=args.evidence_max_chars,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"Built {len(samples)} judgable samples (have generated_title and original_title).")
    n_with_ev = sum(1 for s in samples if s.user_evidence)
    print(f"  with user_evidence injected: {n_with_ev}/{len(samples)}")
    if not samples:
        print("Nothing to judge. Check predictions: need non-empty generated_text and "
              "either an `original_title` field or an 'Original title:' line in context.",
              file=sys.stderr)
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl)) or "."
    os.makedirs(out_dir, exist_ok=True)
    summary_dir = os.path.dirname(os.path.abspath(args.summary_json)) or "."
    os.makedirs(summary_dir, exist_ok=True)
    usage_path = (args.usage_jsonl or default_usage_jsonl_path(args.output_jsonl)).strip()
    u_dir = os.path.dirname(os.path.abspath(usage_path)) or "."
    os.makedirs(u_dir, exist_ok=True)

    existing: Set[str] = set()
    if args.skip_existing:
        existing = load_existing_judge_keys(args.output_jsonl)
        print(f"Skip-existing: {len(existing)} keys already in {args.output_jsonl}.")

    if args.extra_body_json.strip():
        extra_body: Optional[Dict[str, Any]] = json.loads(args.extra_body_json)
    else:
        extra_body = {"top_k": 1}

    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)
    print(f"OpenAI base_url: {args.openai_base_url}; judge model: {args.model}")

    file_lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, args.max_concurrency))
    stats_lock = asyncio.Lock()
    stats: Dict[str, Any] = {
        "ok": 0, "err": 0, "skip": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "requests_with_usage": 0,
    }

    async def process_one(s: JudgeSample) -> None:
        k = judge_key(s.item_id, s.sid)
        if args.skip_existing and k in existing:
            async with stats_lock:
                stats["skip"] += 1
            return
        async with sem:
            try:
                norm, meta, usage = await call_judge_with_retry(
                    client=client, model=args.model, sample=s, seed=args.seed,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    top_p=args.top_p, request_timeout=args.request_timeout,
                    max_retries=args.max_retries, base_delay=args.retry_base_delay,
                    extra_body=extra_body,
                )
                row = {
                    "item_id": s.item_id,
                    "sid": list(s.sid),
                    "original_title": s.original_title,
                    "generated_title": s.generated_title,
                    "personalization": norm["personalization"],
                    "hallucination": norm["hallucination"],
                    "fluency": norm["fluency"],
                    "overall": norm["overall"],
                    "strict_win": is_strict_win(norm),
                    "reason": norm["reason"],
                    "judge_meta": {"original_pos": meta["original_pos"]},
                }
                async with file_lock:
                    with open(args.output_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                    existing.add(k)
                    urec = {
                        "ts": time.time(),
                        "item_id": s.item_id,
                        "sid": list(s.sid),
                        "model": args.model,
                        "usage": usage,
                    }
                    with open(usage_path, "a", encoding="utf-8") as uf:
                        uf.write(json.dumps(urec, ensure_ascii=False) + "\n")
                        uf.flush()
                async with stats_lock:
                    stats["ok"] += 1
                    if usage:
                        stats["requests_with_usage"] += 1
                        stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                        stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                        stats["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            except Exception:
                print(f"[ERROR] item={s.item_id} sid={s.sid}:\n{traceback.format_exc()}",
                      file=sys.stderr)
                async with stats_lock:
                    stats["err"] += 1

    async def run_all() -> None:
        if args.no_progress:
            await asyncio.gather(*[process_one(s) for s in samples])
            return
        pbar = tqdm(total=len(samples), desc="LLM-judge", unit="row",
                    file=sys.stdout, mininterval=0.3)

        async def run_one(s: JudgeSample) -> None:
            try:
                await process_one(s)
            finally:
                pbar.update(1)
                pbar.set_postfix(ok=stats["ok"], err=stats["err"], skip=stats["skip"], refresh=False)

        try:
            await asyncio.gather(*[run_one(s) for s in samples])
        finally:
            pbar.close()

    await run_all()

    judgments = load_jsonl(args.output_jsonl)
    summary = summarize(judgments)
    summary["api"] = {
        "successful_calls": stats["ok"],
        "errors": stats["err"],
        "skipped": stats["skip"],
        "requests_with_usage": stats["requests_with_usage"],
        "sum_prompt_tokens": stats["prompt_tokens"],
        "sum_completion_tokens": stats["completion_tokens"],
        "sum_total_tokens": stats["total_tokens"],
        "judge_model": args.model,
    }
    summary["paths"] = {
        "pred_jsonl": args.pred_jsonl,
        "reviews_jsonl": args.reviews_jsonl,
        "user_sid": args.user_sid,
        "output_jsonl": args.output_jsonl,
        "usage_jsonl": usage_path,
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if stats["err"] == 0 else 2


def main() -> None:
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        print("Interrupted. 已写入的 judge 行已保留; 加 --skip-existing 续跑。", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
