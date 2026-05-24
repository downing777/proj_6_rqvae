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

评判方式 (每条样本三次 judge, 并发发起):
  Pass 1 (个性化, A/B 盲比): 只评 personalization 这一维度。
    每条随机决定 original/generated 落 A 或 B, 再把 judge 回答映射回
    (original_better / generated_better / tie)。汇总成 generated 视角的胜率。
    prompt 内显式加了 length-neutrality 规则, 避免 judge 因标题长短产生偏置。
  Pass 2 (幻觉检测, 单向): 给 judge 商品客观信息 + 原始标题 (视为无幻觉的 ground truth)
    + 生成标题, 判断生成标题是否引入了未被支持的具体声明 (属性/数字/材质/认证等),
    输出 has_hallucination (bool) + hallucinated_claims (list[str])。
    汇总成 hallucination rate (有幻觉的生成标题占比) —— 不是胜率。
  Pass 3 (流畅度, 单向): 给 judge 原标题 (作为 reference / fluency 下限) + 生成标题,
    只问"生成标题的流畅度是否明显比原标题下降", 输出 fluency_drop (bool)。
    保守裁决 (旗鼓相当就 false), 汇总成 fluency_not_worse_rate
    (流畅度没退化的样本占比) —— 不是胜率。

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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

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


# 同时兼容 dpo_title_gen 的 "Original title: ..." 和 build_eval_data 的
# "原商品标题(英文): ..." / "原商品标题: ..." (全角/半角冒号都接受)。
# 不修这里, build_eval_data 出来的 context 会让 compare judge 直接看到原标题,
# A/B 盲比就失效了。
_ORIGINAL_TITLE_RE = re.compile(
    r"^\s*(?:Original\s+title|原商品标题(?:\s*\(英文\))?)\s*[:：]\s*(.+)$",
    re.IGNORECASE,
)


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
    """剥掉 context 里的 'Original title:' 行 —— compare judge 不能直接看到原文,
    幻觉 judge 也避免重复 (我们已经把原始标题作为独立字段单独给它)。"""
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
#  LLM judge prompts
# --------------------------------------------------------------------------- #


JUDGE_COMPARE_SYSTEM_PROMPT = (
    "You are a strict, careful evaluator for e-commerce product titles. "
    "Given a product's objective info, real reviews from the target user group, "
    "and two candidate titles (labeled A and B), judge ONE dimension: personalization.\n\n"
    "CRITICAL LENGTH-NEUTRALITY RULE:\n"
    "- Title length itself is NOT a quality signal. Do NOT prefer a title for being longer, "
    "\"more descriptive\", or \"having more keywords\". Do NOT penalize a title for being concise.\n"
    "- A short, focused title can be equally or more personalized compared with a long, "
    "keyword-stuffed one. A long title is not automatically more informative — it may just be "
    "repetitive or noisy.\n"
    "- Two titles of very different length can absolutely be a tie. When in doubt about length, "
    "output tie rather than rewarding the longer one.\n"
    "- Judge ONLY the content quality, regardless of word count or character count.\n\n"
    "personalization: Which title better matches the interests / preferences of the target user group "
    "as revealed by their reviews (highlights features they care about, uses framing that resonates)? "
    "Values: A_better / B_better / tie. If user-group evidence is empty, output tie. "
    "Listing many generic keywords does NOT count as personalization — the highlighted feature must "
    "actually be something the user-group reviews care about.\n\n"
    "Do NOT judge fluency or factual accuracy / hallucination here — those are evaluated in separate passes.\n"
    "Output STRICTLY a single JSON object with fields personalization, reason. "
    "reason is a one-sentence explanation. No additional text."
)

JUDGE_COMPARE_USER_TEMPLATE = (
    "[Product objective info]\n{product_attrs}\n\n"
    "[Target SID user-group evidence]\n{user_evidence}\n\n"
    "[Candidate A]\n{title_a}\n\n"
    "[Candidate B]\n{title_b}\n\n"
    "Output JSON only: "
    "{{\"personalization\": \"A_better|B_better|tie\", "
    "\"reason\": \"...\"}}"
)


JUDGE_HALLU_SYSTEM_PROMPT = (
    "You are a strict, careful fact-checker for e-commerce product titles. "
    "You are given (1) a product's objective info, (2) the ORIGINAL ground-truth title "
    "(authoritative, treated as containing no hallucinations), and (3) a NEW generated title to check.\n"
    "Your task: decide whether the NEW title introduces any SPECIFIC, VERIFIABLE claim "
    "(attribute / feature / material / number / size / capacity / certification / brand / model / "
    "compatibility / ingredient) that is NOT supported by either the product info or the original title.\n"
    "Rules:\n"
    "- Subjective adjectives ('stylish', 'comfortable', 'premium-feeling', 'great for ...') are NOT hallucinations.\n"
    "- Reordering, paraphrasing, dropping or shortening information is NOT a hallucination.\n"
    "- Re-using or rephrasing claims already present in the original title or product info is NOT a hallucination.\n"
    "- Adding a specific numeric spec (e.g. '20-hour battery', '4K'), material ('genuine leather'), "
    "certification ('FDA-approved'), brand, model number, or concrete feature not supported by either "
    "the original title or the product info IS a hallucination.\n"
    "- If unsure whether a claim is supported, treat it as NOT a hallucination (be conservative).\n"
    "Output STRICTLY a single JSON object with fields has_hallucination (true|false), "
    "hallucinated_claims (array of short strings quoting the unsupported claims; empty array if none), "
    "reason (one-sentence explanation). No additional text."
)

JUDGE_HALLU_USER_TEMPLATE = (
    "[Product objective info]\n{product_attrs}\n\n"
    "[Original title (ground truth, no hallucination)]\n{original_title}\n\n"
    "[New generated title to check]\n{generated_title}\n\n"
    "Output JSON only: "
    "{{\"has_hallucination\": true|false, "
    "\"hallucinated_claims\": [\"...\"], "
    "\"reason\": \"...\"}}"
)


JUDGE_FLUENCY_SYSTEM_PROMPT = (
    "You are a strict English-fluency reviewer for e-commerce product titles. "
    "You will be given a REFERENCE title (the original product title, treated as the fluency floor) "
    "and a CANDIDATE title (a newly generated title to check). "
    "Your ONLY task: decide whether the CANDIDATE is meaningfully LESS FLUENT than the REFERENCE "
    "as an English product title. You are NOT asked which is 'better' or 'more attractive'.\n\n"
    "What counts as a fluency drop (answer true):\n"
    "- Broken grammar or word order in the CANDIDATE that the REFERENCE does not have.\n"
    "- Repeated tokens, gibberish, garbled or non-English text in the CANDIDATE.\n"
    "- Awkward concatenations that read clearly worse than the REFERENCE.\n"
    "- A truncation that leaves a dangling word, half-word, or unfinished phrase.\n\n"
    "What does NOT count as a fluency drop (answer false):\n"
    "- The CANDIDATE being shorter (or longer) than the REFERENCE. Length is NOT fluency.\n"
    "- Different word choice, paraphrasing, or reordering, as long as the result still reads naturally.\n"
    "- Title-case vs. sentence-case differences.\n"
    "- Dropping or omitting information that was in the REFERENCE (that is not a fluency issue).\n"
    "- The CANDIDATE adding new content, as long as it reads naturally (factuality is judged elsewhere).\n\n"
    "Be CONSERVATIVE: if the two read at roughly comparable fluency, answer false. "
    "Only answer true when the CANDIDATE has a clear, identifiable fluency problem the REFERENCE does not have.\n"
    "Output STRICTLY a single JSON object with fields fluency_drop (true|false), reason (one sentence). "
    "No additional text."
)

JUDGE_FLUENCY_USER_TEMPLATE = (
    "[Reference title (original, fluency floor)]\n{original_title}\n\n"
    "[Candidate title (to check)]\n{generated_title}\n\n"
    "Output JSON only: "
    "{{\"fluency_drop\": true|false, "
    "\"reason\": \"...\"}}"
)


# --------------------------------------------------------------------------- #
#  Response parsing / normalization
# --------------------------------------------------------------------------- #


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


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1", "y", "t"):
            return True
    return False


# 现在 compare pass 只剩 personalization 一个维度。
# fluency 改成单向 not-worse 检查 (见 normalize_fluency), 不再走 A/B 比较;
# hallucination 一直就是单向检查。
_COMPARE_DIMS: Tuple[str, ...] = ("personalization",)


def normalize_compare(raw: Dict[str, Any], original_pos: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        k: _map_two(str(raw.get(k, "tie")), original_pos) for k in _COMPARE_DIMS
    }
    out["reason"] = str(raw.get("reason", ""))[:500]
    return out


def normalize_hallucination(raw: Dict[str, Any]) -> Dict[str, Any]:
    claims_raw = raw.get("hallucinated_claims", [])
    if isinstance(claims_raw, list):
        claims = [str(c)[:200] for c in claims_raw if c]
    elif isinstance(claims_raw, str):
        claims = [claims_raw[:200]] if claims_raw.strip() else []
    else:
        claims = []
    has = _to_bool(raw.get("has_hallucination", False))
    # 若声称无幻觉但列出了 claims, 以列表为准 (避免模型出尔反尔)
    if claims and not has:
        has = True
    return {
        "has_hallucination": has,
        "hallucinated_claims": claims,
        "reason": str(raw.get("reason", ""))[:500],
    }


def normalize_fluency(raw: Dict[str, Any]) -> Dict[str, Any]:
    """fluency_drop=True 表示生成标题比原标题流畅度明显下降; 否则视为 not_worse。
    我们对外暴露 fluency_not_worse (= not fluency_drop), 语义更直接。"""
    drop = _to_bool(raw.get("fluency_drop", False))
    return {
        "fluency_drop": drop,
        "fluency_not_worse": (not drop),
        "reason": str(raw.get("reason", ""))[:500],
    }


def is_strict_win(row: Dict[str, Any]) -> bool:
    """严格胜出: 无幻觉 + 流畅度没下降 + personalization 上 generated 更好。"""
    if row.get("has_hallucination"):
        return False
    if not row.get("fluency_not_worse", True):
        return False
    return row.get("personalization") == "generated_better"


# --------------------------------------------------------------------------- #
#  LLM judge calls
# --------------------------------------------------------------------------- #


async def _chat_json(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
    return parse_json_object(raw), extract_usage(resp)


async def _with_retry(
    fn: Callable[[], Awaitable[Any]],
    max_retries: int,
    base_delay: float,
) -> Any:
    last: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except (APIError, APIConnectionError, APITimeoutError, json.JSONDecodeError, ValueError) as e:
            last = e
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3 * base_delay)
            await asyncio.sleep(delay)
    assert last is not None
    raise last


async def judge_compare(
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
    title_a, title_b, original_pos = position_role(sample, seed)
    user_msg = JUDGE_COMPARE_USER_TEMPLATE.format(
        product_attrs=context_without_original_title(sample.context) or "(none)",
        user_evidence=sample.user_evidence or "(none)",
        title_a=title_a,
        title_b=title_b,
    )

    async def _do() -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        return await _chat_json(
            client, model, JUDGE_COMPARE_SYSTEM_PROMPT, user_msg,
            max_tokens, temperature, top_p, request_timeout, extra_body,
        )

    parsed, usage = await _with_retry(_do, max_retries, base_delay)
    norm = normalize_compare(parsed, original_pos)
    meta = {"original_pos": original_pos, "raw_judge": parsed}
    return norm, meta, usage


async def judge_hallucination(
    client: AsyncOpenAI,
    model: str,
    sample: JudgeSample,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    max_retries: int,
    base_delay: float,
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, int]]]:
    user_msg = JUDGE_HALLU_USER_TEMPLATE.format(
        product_attrs=context_without_original_title(sample.context) or "(none)",
        original_title=sample.original_title,
        generated_title=sample.generated_title,
    )

    async def _do() -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        return await _chat_json(
            client, model, JUDGE_HALLU_SYSTEM_PROMPT, user_msg,
            max_tokens, temperature, top_p, request_timeout, extra_body,
        )

    parsed, usage = await _with_retry(_do, max_retries, base_delay)
    norm = normalize_hallucination(parsed)
    meta = {"raw_judge": parsed}
    return norm, meta, usage


async def judge_fluency(
    client: AsyncOpenAI,
    model: str,
    sample: JudgeSample,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    max_retries: int,
    base_delay: float,
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, int]]]:
    user_msg = JUDGE_FLUENCY_USER_TEMPLATE.format(
        original_title=sample.original_title,
        generated_title=sample.generated_title,
    )

    async def _do() -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        return await _chat_json(
            client, model, JUDGE_FLUENCY_SYSTEM_PROMPT, user_msg,
            max_tokens, temperature, top_p, request_timeout, extra_body,
        )

    parsed, usage = await _with_retry(_do, max_retries, base_delay)
    norm = normalize_fluency(parsed)
    meta = {"raw_judge": parsed}
    return norm, meta, usage


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
    """三个维度三种口径:
      - personalization: A/B 盲比, 报 generated 视角的胜率/平局率/败率;
      - fluency: 单向 not-worse 检查, 报 fluency_not_worse_rate (没退化的样本占比);
      - hallucination: 单向幻觉检测, 报 hallucination_rate (有幻觉的样本占比)。
    胜率仅适用于 personalization, fluency/hallucination 不再有"胜率"概念。"""
    n = len(judgments)
    if n == 0:
        return {"sample_count": 0}

    per_dim: Dict[str, Dict[str, Any]] = {}
    for dim in _COMPARE_DIMS:
        gw = sum(1 for j in judgments if j.get(dim) == "generated_better")
        tw = sum(1 for j in judgments if j.get(dim) == "tie")
        ow = sum(1 for j in judgments if j.get(dim) == "original_better")
        per_dim[dim] = {
            "generated_win_rate": gw / n,
            "tie_rate": tw / n,
            "original_win_rate": ow / n,
            "wilson95_generated_win_rate": wilson_ci(gw, n),
        }

    fluency_checked = sum(1 for j in judgments if "fluency_not_worse" in j)
    fluency_not_worse = sum(1 for j in judgments if j.get("fluency_not_worse"))
    fluency_drop = sum(1 for j in judgments
                       if "fluency_not_worse" in j and not j.get("fluency_not_worse"))
    fluency_nw_rate = (fluency_not_worse / fluency_checked) if fluency_checked > 0 else None
    fluency_nw_ci = wilson_ci(fluency_not_worse, fluency_checked) if fluency_checked > 0 else None

    hallu_checked = sum(1 for j in judgments if "has_hallucination" in j)
    hallu_count = sum(1 for j in judgments if j.get("has_hallucination"))
    hallu_rate = (hallu_count / hallu_checked) if hallu_checked > 0 else None
    hallu_ci = wilson_ci(hallu_count, hallu_checked) if hallu_checked > 0 else None

    strict_wins = sum(1 for j in judgments if is_strict_win(j))
    sid_buckets = Counter(",".join(str(x) for x in j["sid"]) for j in judgments)

    return {
        "sample_count": n,
        # generated 视角的胜率, 现在只剩 personalization
        "win_rates": {
            "personalization": per_dim["personalization"]["generated_win_rate"],
        },
        "per_dimension": per_dim,
        # fluency: 单向 not-worse 检查 (没退化的占比), 不是胜率
        "fluency": {
            "checked": fluency_checked,
            "not_worse_count": fluency_not_worse,
            "drop_count": fluency_drop,
            "not_worse_rate": fluency_nw_rate,
            "wilson95_not_worse_rate": fluency_nw_ci,
        },
        # 幻觉率: 生成标题中包含未支持声明的比例 (非胜率)
        "hallucination": {
            "checked": hallu_checked,
            "hallucination_count": hallu_count,
            "hallucination_rate": hallu_rate,
            "wilson95_hallucination_rate": hallu_ci,
        },
        # 严格胜出: 无幻觉 + 流畅度没下降 + personalization 上 generated 更好
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
    ap.add_argument("--max-concurrency", type=int, default=4,
                    help="并发处理的样本数; 每条样本会并发发起 3 次 judge 请求 "
                         "(compare/hallucination/fluency), 因此对 vLLM 的实际并发可达 3x。")
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
                # 三个 judge pass 并发:
                #   compare (personalization, A/B 盲比)
                #   hallucination (单向, 检查生成标题有无未支持声明)
                #   fluency (单向, 检查生成标题流畅度是否比原标题下降)
                cmp_task = judge_compare(
                    client=client, model=args.model, sample=s, seed=args.seed,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    top_p=args.top_p, request_timeout=args.request_timeout,
                    max_retries=args.max_retries, base_delay=args.retry_base_delay,
                    extra_body=extra_body,
                )
                hal_task = judge_hallucination(
                    client=client, model=args.model, sample=s,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    top_p=args.top_p, request_timeout=args.request_timeout,
                    max_retries=args.max_retries, base_delay=args.retry_base_delay,
                    extra_body=extra_body,
                )
                flu_task = judge_fluency(
                    client=client, model=args.model, sample=s,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    top_p=args.top_p, request_timeout=args.request_timeout,
                    max_retries=args.max_retries, base_delay=args.retry_base_delay,
                    extra_body=extra_body,
                )
                (cmp_norm, cmp_meta, cmp_usage), \
                (hal_norm, hal_meta, hal_usage), \
                (flu_norm, flu_meta, flu_usage) = \
                    await asyncio.gather(cmp_task, hal_task, flu_task)
                row: Dict[str, Any] = {
                    "item_id": s.item_id,
                    "sid": list(s.sid),
                    "original_title": s.original_title,
                    "generated_title": s.generated_title,
                    "personalization": cmp_norm["personalization"],
                    "fluency_not_worse": flu_norm["fluency_not_worse"],
                    "fluency_drop": flu_norm["fluency_drop"],
                    "has_hallucination": hal_norm["has_hallucination"],
                    "hallucinated_claims": hal_norm["hallucinated_claims"],
                    "reason_compare": cmp_norm["reason"],
                    "reason_fluency": flu_norm["reason"],
                    "reason_hallucination": hal_norm["reason"],
                    "judge_meta": {"original_pos": cmp_meta["original_pos"]},
                }
                row["strict_win"] = is_strict_win(row)
                async with file_lock:
                    with open(args.output_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                    existing.add(k)
                    with open(usage_path, "a", encoding="utf-8") as uf:
                        for stage, usage in (
                            ("compare", cmp_usage),
                            ("hallucination", hal_usage),
                            ("fluency", flu_usage),
                        ):
                            urec = {
                                "ts": time.time(),
                                "item_id": s.item_id,
                                "sid": list(s.sid),
                                "model": args.model,
                                "stage": stage,
                                "usage": usage,
                            }
                            uf.write(json.dumps(urec, ensure_ascii=False) + "\n")
                        uf.flush()
                async with stats_lock:
                    stats["ok"] += 1
                    for usage in (cmp_usage, hal_usage, flu_usage):
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
        "calls_per_sample": 3,
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
