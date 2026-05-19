#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
就地清洗已生成的 JSONL: 从每行的 `context` 字段里抠掉
`Target SID (rqid_0, rqid_1, rqid_2): [x, y, z]` 这一行。

背景: SID 不应以文本形式出现在 prompt 里, 否则学生模型会形成 shortcut,
导致 SidPrefixEncoder (soft prefix) 学不到东西。

用法:
  python3 softprompt/data/modify_data.py FILE1 [FILE2 ...] [--no-backup] [--dry-run]

默认会在原文件旁生成 `<file>.bak` 备份, 然后原地覆盖。--dry-run 不写盘只打印统计。
对没有命中目标行的文件也安全 (会报告 modified=0)。

典型清洗目标 (按 run_train.sh / run_dpo_title_gen.sh 的产物):
  ${OUT_DIR}/data/dpo_electronics_generated_<MODEL>.jsonl
  ${OUT_DIR}/data/dpo_electronics_generated_<MODEL>.titles_cache.jsonl
  ${OUT_DIR}/split/train.jsonl
  ${OUT_DIR}/split/test.jsonl
  ${OUT_DIR}/split/test_infer.jsonl
  ${OUT_DIR}/sft_from_item_title.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import Tuple

# 匹配整行 (含末尾换行)。允许 `[26, 22, 7]` / `[26,22,7]` 等空格变体。
SID_LINE_RE = re.compile(
    r"^Target SID \(rqid_0, rqid_1, rqid_2\):[^\n]*\n?",
    flags=re.MULTILINE,
)


def strip_sid_line(text: str) -> Tuple[str, int]:
    """返回 (清洗后字符串, 被删除的行数)。"""
    if not isinstance(text, str) or "Target SID" not in text:
        return text, 0
    new_text, n = SID_LINE_RE.subn("", text)
    # 末尾可能多出一个空行 (原行没带 \n 时)
    new_text = new_text.rstrip("\n") if not text.endswith("\n") else new_text
    return new_text, n


def process_file(path: str, dry_run: bool, no_backup: bool) -> None:
    if not os.path.isfile(path):
        print(f"[skip] not a file: {path}", file=sys.stderr)
        return

    total_rows = 0
    modified_rows = 0
    stripped_lines = 0
    bad_json = 0
    no_context = 0

    cleaned_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw_stripped = raw.strip()
            if not raw_stripped:
                cleaned_lines.append(raw)
                continue
            total_rows += 1
            try:
                row = json.loads(raw_stripped)
            except json.JSONDecodeError:
                bad_json += 1
                cleaned_lines.append(raw)  # 原样保留, 不破坏未知格式
                continue
            ctx = row.get("context")
            if ctx is None:
                no_context += 1
                cleaned_lines.append(raw)
                continue
            new_ctx, n = strip_sid_line(ctx)
            if n > 0:
                row["context"] = new_ctx
                modified_rows += 1
                stripped_lines += n
            cleaned_lines.append(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[{path}] rows={total_rows} modified={modified_rows} "
        f"stripped_lines={stripped_lines} bad_json={bad_json} no_context={no_context}"
    )

    if dry_run:
        print("  (dry-run: 未写盘)")
        return
    if modified_rows == 0:
        print("  (无需改动, 跳过写盘)")
        return

    if not no_backup:
        bak_path = path + ".bak"
        if os.path.exists(bak_path):
            print(f"  WARNING: 备份已存在, 覆盖: {bak_path}", file=sys.stderr)
        shutil.copy2(path, bak_path)
        print(f"  备份 -> {bak_path}")

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)
    os.replace(tmp_path, path)
    print(f"  写回 -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="要清洗的 JSONL 文件路径(可多个)")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计, 不写盘")
    ap.add_argument("--no-backup", action="store_true", help="跳过 .bak 备份 (危险)")
    args = ap.parse_args()

    for p in args.paths:
        process_file(p, dry_run=args.dry_run, no_backup=args.no_backup)


if __name__ == "__main__":
    main()
