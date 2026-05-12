#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-judge offline evaluation for SID-conditioned title generation.

输入:
  --pred-jsonl: softprompt/infer/generate_title.py 的产物, 每行含
      {item_id, sid, context, generated_text}
    其中 generated_text 通常是 "<prompt>...<生成的标题>", 本脚本会剥离 prompt 前缀。

  --dpo-jsonl (可选): softprompt/dpo_title_gen.py 的产物, 每行含
      {item_id, sid, context, title_chosen, title_rejected, ...}
    若提供, 当 prediction.context 缺失时用此处 context 兜底; 同时把 title_chosen
    作为参考 (仅用于产出报表, 不参与 judge 决策)。

评判维度 (LLM-as-judge):
  1) personalization: 相比原标题, 新标题对当前 SID 用户群是否更个性化 (better/same/worse)
  2) hallucination:   新标题是否引入了商品上下文不存在的属性/卖点 (none/minor/major)
  3) fluency:         相比原标题, 新标题的中文通顺度 (better/same/worse)
  4) overall:         综合上面三点, 新标题相对原标题 (win/tie/lose)

判定 "新标题更好" 的口径 (用于胜率统计):
  - personalization in {better}
  - hallucination in {none, minor}
  - fluency in {better, same}
  - overall == win
任一不满足则视作非胜出 (tie / lose 计入对应桶)。

为消除 judge 的位置偏置, 每条样本随机决定原标题/新标题是 A 还是 B,
然后再把 judge 的回答映射回 (original, generated)。

并发/重试/usage 记录/skip-existing 的实现风格沿用 dpo_title_gen.py。

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


DEFAULT_OPENAI_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OPENAI_API_KEY = "EMPTY"

SidTuple = Tuple[int, ...]


# --------------------------------------------------------------------------- #
#  Data loading & cleaning
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


_PROMPT_TAIL = "请为该 SID 用户群生成一个吸引点击的商品标题："


def strip_prompt_prefix(generated_text: str, context: Optional[str]) -> str:
    """generate_title.py 不剥离 prompt, 这里把 prompt 部分去掉, 仅保留模型续写。"""
    if not generated_text:
        return ""
    text = generated_text
    idx = text.rfind(_PROMPT_TAIL)
    if idx >= 0:
        text = text[idx + len(_PROMPT_TAIL):]
    elif context:
        ctx_idx = text.rfind(context)
        if ctx_idx >= 0:
            text = text[ctx_idx + len(context):]
            tail_idx = text.find(_PROMPT_TAIL)
            if tail_idx >= 0:
                text = text[tail_idx + len(_PROMPT_TAIL):]
    text = text.strip().lstrip("：:").strip()
    text = re.sub(r"\s+", " ", text)
    return text


_TITLE_PATTERNS = [
    re.compile(r"原商品标题\(英文\):\s*(.+)"),
    re.compile(r"原商品标题:\s*(.+)"),
    re.compile(r"商品标题:\s*(.+)"),
]


def extract_original_title(context: str) -> str:
    if not context:
        return ""
    for pat in _TITLE_PATTERNS:
        for line in context.splitlines():
            m = pat.search(line.strip())
            if m:
                return m.group(1).strip().strip("。.")
    return ""


_SID_LINE = re.compile(r"目标\s*SID[^:：]*[:：]\s*(\[[^\]]*\])")


def context_user_evidence(context: str) -> str:
    """从 dpo_title_gen 风格的 context 里抽取 '该 SID 用户群的评论证据' 段, 给 judge 用作用户画像。"""
    if not context:
        return ""
    lines = context.splitlines()
    keep: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("该 SID 用户群") or s.startswith("同商品另一 SID") or s.startswith("目标 SID"):
            keep.append(s)
    return "\n".join(keep)


def context_product_attrs(context: str) -> str:
    """商品属性段 (除评论证据/SID 外的客观信息), 给 judge 判断幻觉。"""
    if not context:
        return ""
    out: List[str] = []
    for line in context.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("该 SID 用户群") or s.startswith("同商品另一 SID") or s.startswith("目标 SID"):
            continue
        out.append(s)
    return "\n".join(out)


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
    reference_chosen: Optional[str]  # 仅作报表参考, 不告知 judge


def build_samples(
    pred_rows: List[Dict[str, Any]],
    dpo_rows: List[Dict[str, Any]],
    fallback_original_title: str = "",
) -> List[JudgeSample]:
    dpo_map: Dict[Tuple[str, SidTuple], Dict[str, Any]] = {}
    for r in dpo_rows:
        sid = to_sid_tuple(r.get("sid"))
        iid = r.get("item_id")
        if iid and sid:
            dpo_map[(str(iid), sid)] = r

    samples: List[JudgeSample] = []
    for r in pred_rows:
        sid = to_sid_tuple(r.get("sid"))
        iid = r.get("item_id")
        if not iid or sid is None:
            continue
        context = r.get("context") or ""
        dpo_row = dpo_map.get((str(iid), sid))
        if not context and dpo_row is not None:
            context = dpo_row.get("context") or ""
        gen_text = r.get("generated_text") or ""
        gen_title = strip_prompt_prefix(gen_text, context)
        if not gen_title:
            continue
        original = extract_original_title(context) or fallback_original_title
        if not original:
            continue
        ref_chosen = None
        if dpo_row is not None:
            tc = dpo_row.get("title_chosen")
            if isinstance(tc, str) and tc.strip():
                ref_chosen = tc.strip()
        samples.append(
            JudgeSample(
                item_id=str(iid),
                sid=sid,
                context=context,
                original_title=original,
                generated_title=gen_title,
                reference_chosen=ref_chosen,
            )
        )
    return samples


# --------------------------------------------------------------------------- #
#  LLM judge
# --------------------------------------------------------------------------- #


JUDGE_SYSTEM_PROMPT = (
    "你是严格、谨慎的电商标题评测员。给定一个商品的客观属性、目标 SID 用户群的评论证据, "
    "以及该商品的两个候选中文标题(标记为 A 与 B), 请按以下三个维度做对比, 然后给出综合偏好。\n"
    "维度:\n"
    "1) personalization: 相对另一个标题, 候选标题是否更贴合 [目标 SID 用户群] 在评论证据里"
    "暴露出的关注点/偏好。取值: A_better / B_better / tie。\n"
    "2) hallucination: 候选标题是否引入了 [商品客观属性] 中找不到依据的具体属性/功能/材质/数字 (如不存在的"
    "续航时长、不存在的认证、夸大的型号差异等)。取值: A_better / B_better / tie\n"
    "   (better 含义为该候选幻觉更少; 若两者都几乎无幻觉则 tie)。\n"
    "3) fluency: 中文表达是否通顺自然 (语序、搭配、是否有重复字符或乱码)。"
    "取值: A_better / B_better / tie。\n"
    "4) overall: 综合上面三点, 给出整体偏好 (win 给更好的那个候选)。"
    "取值: A_win / B_win / tie。\n"
    "判罚规则: 如果一个候选包含明显的幻觉/事实错误, overall 不能判它 win。"
    "如果一个候选有重复字符/乱码/明显不通顺, overall 不能判它 win。\n"
    "请严格只输出一个 JSON 对象, 字段为 personalization, hallucination, fluency, overall, reason。"
    "其中 reason 用一句话简述。不要任何额外文本。"
)

JUDGE_USER_TEMPLATE = (
    "[商品客观属性]\n{product_attrs}\n\n"
    "[目标 SID 用户群证据]\n{user_evidence}\n\n"
    "[候选 A]\n{title_a}\n\n"
    "[候选 B]\n{title_b}\n\n"
    "请只输出 JSON: "
    "{{\"personalization\": \"A_better|B_better|tie\", "
    "\"hallucination\": \"A_better|B_better|tie\", "
    "\"fluency\": \"A_better|B_better|tie\", "
    "\"overall\": \"A_win|B_win|tie\", "
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
    """决定原标题/生成标题各自落到 A 还是 B (位置随机化)。返回 (title_a, title_b, original_pos)。"""
    h = int(hashlib.md5(f"{sample.item_id}:{sample.sid}:{seed}".encode()).hexdigest()[:8], 16)
    if h % 2 == 0:
        return sample.original_title, sample.generated_title, "A"
    return sample.generated_title, sample.original_title, "B"


def _map_two(value: str, original_pos: str) -> str:
    """把 A_better/B_better/tie 翻译成 original_better/generated_better/tie。"""
    v = (value or "").strip().lower()
    if v not in ("a_better", "b_better", "tie"):
        return "tie"
    if v == "tie":
        return "tie"
    chose = "A" if v == "a_better" else "B"
    return "original_better" if chose == original_pos else "generated_better"


def _map_overall(value: str, original_pos: str) -> str:
    """A_win/B_win/tie -> original_win/generated_win/tie。"""
    v = (value or "").strip().lower()
    if v == "tie":
        return "tie"
    if v not in ("a_win", "b_win"):
        return "tie"
    chose = "A" if v == "a_win" else "B"
    return "original_win" if chose == original_pos else "generated_win"


def normalize_judge(raw: Dict[str, Any], original_pos: str) -> Dict[str, Any]:
    return {
        "personalization": _map_two(str(raw.get("personalization", "tie")), original_pos),
        "hallucination": _map_two(str(raw.get("hallucination", "tie")), original_pos),
        "fluency": _map_two(str(raw.get("fluency", "tie")), original_pos),
        "overall": _map_overall(str(raw.get("overall", "tie")), original_pos),
        "reason": str(raw.get("reason", ""))[:500],
    }


def is_strict_win(norm: Dict[str, Any]) -> bool:
    return (
        norm["personalization"] == "generated_better"
        and norm["hallucination"] in ("generated_better", "tie")
        and norm["fluency"] in ("generated_better", "tie")
        and norm["overall"] == "generated_win"
    )


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
        product_attrs=context_product_attrs(sample.context) or "(无)",
        user_evidence=context_user_evidence(sample.context) or "(无)",
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
    n = len(judgments)
    if n == 0:
        return {"sample_count": 0}

    def rate(field: str, target: str) -> float:
        return sum(1 for j in judgments if j[field] == target) / n

    overall_counts = Counter(j["overall"] for j in judgments)
    strict_wins = sum(1 for j in judgments if is_strict_win(j))

    overall_win = overall_counts.get("generated_win", 0)
    overall_tie = overall_counts.get("tie", 0)
    overall_lose = overall_counts.get("original_win", 0)

    # 控制 tie 后的胜率 (排除 tie 后, 生成 vs 原始 的胜出比例)
    decisive = overall_win + overall_lose
    win_rate_excl_tie = (overall_win / decisive) if decisive > 0 else None

    sid_buckets = Counter(",".join(str(x) for x in j["sid"]) for j in judgments)

    summary: Dict[str, Any] = {
        "sample_count": n,
        "overall": {
            "generated_win_rate": overall_win / n,
            "tie_rate": overall_tie / n,
            "original_win_rate": overall_lose / n,
            "win_rate_excluding_tie": win_rate_excl_tie,
            "wilson95_generated_win_rate": wilson_ci(overall_win, n),
        },
        "strict_win_rate": strict_wins / n,
        "wilson95_strict_win_rate": wilson_ci(strict_wins, n),
        "personalization": {
            "generated_better": rate("personalization", "generated_better"),
            "tie": rate("personalization", "tie"),
            "original_better": rate("personalization", "original_better"),
        },
        "hallucination": {
            "generated_better": rate("hallucination", "generated_better"),
            "tie": rate("hallucination", "tie"),
            "original_better": rate("hallucination", "original_better"),
            "generated_no_worse_rate": (
                sum(
                    1 for j in judgments
                    if j["hallucination"] in ("generated_better", "tie")
                ) / n
            ),
        },
        "fluency": {
            "generated_better": rate("fluency", "generated_better"),
            "tie": rate("fluency", "tie"),
            "original_better": rate("fluency", "original_better"),
            "generated_no_worse_rate": (
                sum(
                    1 for j in judgments
                    if j["fluency"] in ("generated_better", "tie")
                ) / n
            ),
        },
        "sid_bucket_count": len(sid_buckets),
    }
    return summary


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


async def amain() -> int:
    ap = argparse.ArgumentParser(description="LLM-as-judge offline eval for SID title generation.")
    ap.add_argument("--pred-jsonl", type=str, required=True,
                    help="generate_title.py 输出, 每行 {item_id, sid, context, generated_text}")
    ap.add_argument("--dpo-jsonl", type=str, default="",
                    help="可选, dpo_title_gen.py 输出, 用于在 prediction 缺失 context 时兜底")
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
    args = ap.parse_args()

    if not os.path.isfile(args.pred_jsonl):
        print(f"Error: --pred-jsonl not found: {args.pred_jsonl}", file=sys.stderr)
        return 1

    pred_rows = load_jsonl(args.pred_jsonl)
    dpo_rows = load_jsonl(args.dpo_jsonl) if args.dpo_jsonl and os.path.isfile(args.dpo_jsonl) else []
    print(f"Loaded {len(pred_rows)} predictions; {len(dpo_rows)} DPO rows for context fallback.")

    samples = build_samples(pred_rows, dpo_rows)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"Built {len(samples)} judgable samples (have generated_title and original_title).")
    if not samples:
        print("Nothing to judge. Check that generated_text is non-empty and context contains "
              "'原商品标题(英文): ...' line.", file=sys.stderr)
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
                    "reference_chosen": s.reference_chosen,
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
        pbar = tqdm(total=len(samples), desc="LLM-judge", unit="条",
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
        "dpo_jsonl": args.dpo_jsonl,
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
        print("Interrupted. 已写入的 judge 行已保留; 可加 --skip-existing 续跑。", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
