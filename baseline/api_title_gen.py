#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline: 直接调一个 OpenAI 兼容的 chat completion API, 让 LLM 根据
"商品客观信息 + 该 SID 用户群在本商品下的评论" 直接生成一条个性化短标题。
不训练, 不做软提示, 纯粹是 zero-shot prompt → 文本。

输入 (与 softprompt.infer.generate_title 同口径):
  --input-jsonl: split_data.py 产出的 test_infer.jsonl, 每行至少含
      {item_id, sid, context, ...} (可选 original_title / title_chosen / user_id)
      context 由 dpo_title_gen.py 构造, 既有商品属性也有 SID 用户群评论;
      其中 "Original title: ..." 这一行会被剥掉, 避免基线模型直接 copy 原标题。
      没有 original_title 字段时, 会从 context 里 fallback 抽 "Original title:" 行,
      跟 generate_title.py 完全一致。

输出 (兼容 softprompt/eval/offline_eval.py):
  --output-jsonl: 每行 {item_id, sid, context, generated_text, original_title}
      —— 这与 generate_title.py 的输出 schema 完全一致, 可以直接喂给
      offline_eval.py 做 LLM-as-judge。

依赖: pip install openai tqdm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import traceback
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



SidTuple = Tuple[int, ...]


# --------------------------------------------------------------------------- #
#  Data loading / context cleaning
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


# 与 softprompt/eval/offline_eval.py 同口径, 既支持英文标签也支持中文标签 +
# 全角/半角冒号; 必须剥掉, 否则基线模型大概率直接抄原标题。
_ORIGINAL_TITLE_RE = re.compile(
    r"^\s*(?:Original\s+title|原商品标题(?:\s*\(英文\))?)\s*[:：]\s*(.+)$",
    re.IGNORECASE,
)


def extract_original_title(context: str) -> str:
    if not context:
        return ""
    for line in context.splitlines():
        m = _ORIGINAL_TITLE_RE.match(line)
        if m:
            return m.group(1).strip().strip(".")
    return ""


def context_without_original_title(context: str) -> str:
    if not context:
        return ""
    kept = [line for line in context.splitlines() if not _ORIGINAL_TITLE_RE.match(line)]
    return "\n".join(kept).strip()


# --------------------------------------------------------------------------- #
#  Prompt
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert e-commerce copywriter for an Amazon-style marketplace. "
    "You will be given a product's objective information (category, brand, price, "
    "feature bullets, etc.) plus REAL reviews written by a specific target user group "
    "(identified by a semantic-ID cluster). Your job is to write ONE new, personalized "
    "English product title for this product, tailored to that user group.\n\n"
    "Hard requirements:\n"
    "- The title must be in ENGLISH only. No Chinese, no emojis.\n"
    "- Surface ONE or TWO concrete aspects of the product that the target user group's "
    "reviews actually care about (e.g. a feature they praise or rely on). Do NOT just "
    "dump generic SEO keywords.\n"
    "- Do NOT invent specific attributes, materials, numbers, certifications, brand / "
    "model names, or features that are not supported by the product info or reviews. "
    "If you are unsure whether a claim is supported, leave it out.\n"
    "- Be CONCISE: aim for ~{target_words} words, single line, Title Case. No terminal "
    "period, no quotes, no markdown, no leading dash or bullet.\n"
    "- Read naturally as a product title, not as a sentence and not as a keyword salad.\n\n"
    "Output ONLY the title text itself. No preamble, no explanation, no \"Title:\" prefix, "
    "no surrounding quotes."
)


USER_PROMPT_TEMPLATE = (
    "[Product information and target user-group reviews]\n"
    "{context}\n\n"
    "Write the personalized title now. Output only the title, no other text."
)


def build_prompts(context: str, target_words: int) -> Tuple[str, str]:
    sys_p = SYSTEM_PROMPT_TEMPLATE.format(target_words=target_words)
    cleaned = context_without_original_title(context) or "(none)"
    usr_p = USER_PROMPT_TEMPLATE.format(context=cleaned)
    return sys_p, usr_p


# --------------------------------------------------------------------------- #
#  Output post-processing
# --------------------------------------------------------------------------- #


_TITLE_PREFIX_RE = re.compile(r"^\s*(?:title|product\s+title|new\s+title)\s*[:：]\s*", re.IGNORECASE)
_LEADING_BULLET_RE = re.compile(r"^[\s\-\*•]+")


def clean_generated_title(text: str) -> str:
    """剥掉模型偶尔会加的 'Title: ', 反引号, 引号, 多余换行等 wrapping。
    保留原文本身的措辞 —— 不做语义改写。"""
    if not text:
        return ""
    s = text.strip()
    # 去掉首尾代码块标记
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()
    # 只保留第一行 (模型偶尔会写一行标题 + 一行解释)
    first = s.splitlines()[0].strip() if s else ""
    if not first:
        return ""
    # 去掉 'Title: ' 一类的前缀
    first = _TITLE_PREFIX_RE.sub("", first).strip()
    # 去掉行首的 bullet/dash
    first = _LEADING_BULLET_RE.sub("", first).strip()
    # 去掉首尾匹配的引号
    if len(first) >= 2 and first[0] == first[-1] and first[0] in ('"', "'", "“", "”", "‘", "’", "`"):
        first = first[1:-1].strip()
    # 去掉结尾的句号
    first = first.rstrip(".。 ").strip()
    return first


# --------------------------------------------------------------------------- #
#  API call
# --------------------------------------------------------------------------- #


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


async def _chat_text(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[str, Optional[Dict[str, int]]]:
    # Build messages: if max_tokens <= 0 (minimal-params mode), merge system into user
    # to avoid providers that reject system role
    if max_tokens <= 0:
        merged_content = system_prompt + "\n\n" + user_prompt
        messages = [{"role": "user", "content": merged_content}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    if temperature >= 0:
        kwargs["temperature"] = temperature
    if top_p < 1.0:
        kwargs["top_p"] = top_p
    if request_timeout > 0:
        kwargs["timeout"] = request_timeout
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = await client.chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "")
    if not raw.strip():
        raise ValueError("Empty model response")
    return raw, extract_usage(resp)


async def _chat_text_raw(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, Optional[Dict[str, int]]]:
    """Minimal raw HTTP call — only model+messages, no extra params.
    Used for providers (like idealab) that reject any additional fields."""
    import httpx
    url = base_url.rstrip("/") + "/chat/completions"
    merged_content = system_prompt + "\n\n" + user_prompt
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": merged_content}],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    async with httpx.AsyncClient(timeout=300.0) as http_client:
        resp = await http_client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise ValueError(
                f"HTTP {resp.status_code} from {url}: {resp.text[:500]}"
            )
    data = resp.json()
    raw = data["choices"][0]["message"]["content"] or ""
    if not raw.strip():
        raise ValueError("Empty model response")
    usage = data.get("usage")
    return raw, usage


async def _with_retry(
    fn: Callable[[], Awaitable[Any]],
    max_retries: int,
    base_delay: float,
) -> Any:
    last: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except (APIError, APIConnectionError, APITimeoutError, ValueError, Exception) as e:
            last = e
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3 * base_delay)
            await asyncio.sleep(delay)
    assert last is not None
    raise last


# --------------------------------------------------------------------------- #
#  Resume support (skip-existing)
# --------------------------------------------------------------------------- #


def sample_key(item_id: str, sid: SidTuple) -> str:
    return f"{item_id}::{','.join(str(x) for x in sid)}"


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
            sid = to_sid_tuple(row.get("sid"))
            if iid and sid:
                out.add(sample_key(str(iid), sid))
    return out


def default_usage_jsonl_path(output_jsonl: str) -> str:
    if output_jsonl.endswith(".jsonl"):
        return output_jsonl[: -len(".jsonl")] + ".usage.jsonl"
    return output_jsonl + ".usage.jsonl"


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


async def amain() -> int:
    ap = argparse.ArgumentParser(
        description="Baseline: call an OpenAI-compatible API to generate personalized titles."
    )
    ap.add_argument("--input-jsonl", type=str, required=True,
                    help="split_data.py 产出的 test_infer.jsonl (或任何带 context 的 jsonl)")
    ap.add_argument("--output-jsonl", type=str,
                    default="baseline/outputs/baseline_api_predictions.jsonl",
                    help="预测结果, schema 与 softprompt.infer.generate_title 一致")
    ap.add_argument("--usage-jsonl", type=str, default="",
                    help="token 用量, 默认 <output-jsonl>.usage.jsonl")
    ap.add_argument("--openai-base-url", type=str)
    ap.add_argument("--openai-api-key", type=str)
    ap.add_argument("--model", type=str,
                    help="基线模型 (建议跟被对比的训练模型同级或更强)")
    ap.add_argument("--target-words", type=int, default=12,
                    help="提示模型把标题控制在多少个英文单词左右 (默认 12, 跟训练目标的短标题对齐)")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="生成端 max_new_tokens (标题短, 给个小数就够)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--retry-base-delay", type=float, default=1.0)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="只跑前 N 条 (调试用)")
    ap.add_argument("--seed", type=int, default=42,
                    help="若 --max-samples 启用, 用这个种子做随机抽样")
    ap.add_argument("--random-sample", action=argparse.BooleanOptionalAction, default=False,
                    help="启用 --max-samples 时是否随机抽样, 默认按文件顺序前 N 条")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--extra-body-json", type=str, default="",
                    help="vLLM extra_body JSON; 默认空 (即纯采样)")
    ap.add_argument("--minimal-params", action="store_true", default=False,
                    help="Only send model+messages to the API; skip temperature/top_p/max_tokens "
                         "(for providers like GPT-5 that reject unsupported params)")
    args = ap.parse_args()

    # --minimal-params: 只传 model+messages, 其他参数全部跳过
    if args.minimal_params:
        args.temperature = -1.0       # sentinel: skip
        args.top_p = 1.0              # sentinel: skip
        args.max_tokens = 0           # sentinel: skip
        args.request_timeout = 0.0    # sentinel: skip

    if not os.path.isfile(args.input_jsonl):
        print(f"ERROR: --input-jsonl not found: {args.input_jsonl}", file=sys.stderr)
        return 1

    rows = load_jsonl(args.input_jsonl)
    print(f"Loaded {len(rows)} input rows from {args.input_jsonl}.")

    # Normalize / filter
    samples: List[Dict[str, Any]] = []
    for r in rows:
        iid = r.get("item_id")
        sid = to_sid_tuple(r.get("sid"))
        ctx = r.get("context") or ""
        if not iid or sid is None or not ctx:
            continue
        original = (r.get("original_title") or "").strip() or extract_original_title(ctx)
        samples.append({
            "item_id": str(iid),
            "sid": sid,
            "context": ctx,
            "original_title": original,
        })
    print(f"Usable samples: {len(samples)}")

    if args.max_samples is not None and args.max_samples < len(samples):
        if args.random_sample:
            random.seed(args.seed)
            samples = random.sample(samples, args.max_samples)
            print(f"Randomly sampled {len(samples)} rows (seed={args.seed}).")
        else:
            samples = samples[: args.max_samples]
            print(f"Truncated to first {len(samples)} rows.")

    if not samples:
        print("Nothing to generate.", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl)) or "."
    os.makedirs(out_dir, exist_ok=True)
    usage_path = (args.usage_jsonl or default_usage_jsonl_path(args.output_jsonl)).strip()
    u_dir = os.path.dirname(os.path.abspath(usage_path)) or "."
    os.makedirs(u_dir, exist_ok=True)

    existing: Set[str] = set()
    if args.skip_existing:
        existing = load_existing_keys(args.output_jsonl)
        print(f"Skip-existing: {len(existing)} keys already in {args.output_jsonl}.")

    extra_body: Optional[Dict[str, Any]] = None
    if args.extra_body_json.strip():
        extra_body = json.loads(args.extra_body_json)

    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)
    print(f"OpenAI base_url: {args.openai_base_url}; model: {args.model}; "
          f"target_words={args.target_words}; max_concurrency={args.max_concurrency}")

    file_lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, args.max_concurrency))
    stats_lock = asyncio.Lock()
    stats: Dict[str, int] = {
        "ok": 0, "err": 0, "skip": 0, "empty": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "requests_with_usage": 0,
    }

    async def process_one(s: Dict[str, Any]) -> None:
        k = sample_key(s["item_id"], s["sid"])
        if args.skip_existing and k in existing:
            async with stats_lock:
                stats["skip"] += 1
            return
        sys_p, usr_p = build_prompts(s["context"], args.target_words)
        async with sem:
            try:
                if args.minimal_params:
                    async def _do() -> Tuple[str, Optional[Dict[str, int]]]:
                        return await _chat_text_raw(
                            args.openai_base_url, args.openai_api_key,
                            args.model, sys_p, usr_p,
                        )
                else:
                    async def _do() -> Tuple[str, Optional[Dict[str, int]]]:
                        return await _chat_text(
                            client, args.model, sys_p, usr_p,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            request_timeout=args.request_timeout,
                            extra_body=extra_body,
                        )

                raw, usage = await _with_retry(_do, args.max_retries, args.retry_base_delay)
                title = clean_generated_title(raw)
                if not title:
                    async with stats_lock:
                        stats["empty"] += 1
                row = {
                    "item_id": s["item_id"],
                    "sid": list(s["sid"]),
                    "context": s["context"],
                    "generated_text": title,
                    "original_title": s["original_title"],
                    "raw_model_output": raw.strip(),
                }
                async with file_lock:
                    with open(args.output_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                    existing.add(k)
                    urec = {
                        "ts": time.time(),
                        "item_id": s["item_id"],
                        "sid": list(s["sid"]),
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
                print(f"[ERROR] item={s['item_id']} sid={s['sid']}:\n{traceback.format_exc()}",
                      file=sys.stderr)
                async with stats_lock:
                    stats["err"] += 1

    async def run_all() -> None:
        if args.no_progress:
            await asyncio.gather(*[process_one(s) for s in samples])
            return
        pbar = tqdm(total=len(samples), desc="baseline-gen", unit="row",
                    file=sys.stdout, mininterval=0.3)

        async def run_one(s: Dict[str, Any]) -> None:
            try:
                await process_one(s)
            finally:
                pbar.update(1)
                pbar.set_postfix(ok=stats["ok"], err=stats["err"], skip=stats["skip"],
                                 empty=stats["empty"], refresh=False)

        try:
            await asyncio.gather(*[run_one(s) for s in samples])
        finally:
            pbar.close()

    await run_all()

    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "usage_jsonl": usage_path,
        "model": args.model,
        "target_words": args.target_words,
        "successful_calls": stats["ok"],
        "errors": stats["err"],
        "skipped": stats["skip"],
        "empty_generations": stats["empty"],
        "requests_with_usage": stats["requests_with_usage"],
        "sum_prompt_tokens": stats["prompt_tokens"],
        "sum_completion_tokens": stats["completion_tokens"],
        "sum_total_tokens": stats["total_tokens"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if stats["err"] == 0 else 2


def main() -> None:
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        print("Interrupted. 已写入的行已保留; 加 --skip-existing 续跑。", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
