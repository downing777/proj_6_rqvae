#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排查 from-scratch DPO 训完后输出乱七八糟到底是不是训练 bug。

跑法 (路径已写死):
  cd /home/yuanhanyang.yhy/proj_6_rqvae
  export PYTHONPATH=/home/yuanhanyang.yhy/proj_6_rqvae
  export CUDA_VISIBLE_DEVICES=0
  python3 softprompt/infer/debug_dpo_only.py

会做 3 件事:
  [A] sequence_logp 数值自洽性: 用 SFT-style 的样本算一遍 seq_logp,
      跟 HF base_model 自己算的 out.loss 对比。差 < 1e-3 就证明 DPO 用的
      logp 计算无 bug。
  [B] 比较 "未训过的随机 prefix" vs "DPO 训完的 prefix" 在测试样本上的
      pi_chosen / pi_rejected 数值, 看 DPO 真的把哪个推上去了哪个压下去了。
  [C] DPO ckpt 跟 random init 的实际权重差异, 量化"训了 1000 步到底动了多少"。
"""
import os, sys, json, copy

PROJECT_ROOT = "/home/yuanhanyang.yhy/proj_6_rqvae"
MODEL_PATH   = "/home/yuanhanyang.yhy/model_hub/Qwen3.5-9B"
DPO_CKPT     = "/home/yuanhanyang.yhy/project_6_outputs/weights/dpo_only/sid_dpo.pt"
TRAIN_FILE   = "/home/yuanhanyang.yhy/project_6_outputs/split/train.jsonl"
N_SAMPLES    = 4

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoTokenizer
from softprompt.models import SidModelLoadConfig, build_sid_model
from softprompt.train.common import (
    render_prompt, sequence_logp, build_prompt_target_tensors,
)


def banner(s):
    print()
    print("=" * 72)
    print("  " + s)
    print("=" * 72)


# ---------------------------------------------------------- shared setup ----
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

with open(TRAIN_FILE, "r") as f:
    rows = [json.loads(l) for l in f.read().splitlines() if l.strip()][:N_SAMPLES]

sid     = torch.tensor([r["sid"] for r in rows], dtype=torch.long)
prompts = [render_prompt(r["context"]) for r in rows]
chosens   = [r["title_chosen"]   for r in rows]
rejecteds = [r["title_rejected"] for r in rows]

# ============================================================ [A] 自洽性 ====
banner("[A] sequence_logp self-consistency vs HF base_model.loss")
m_check = build_sid_model(SidModelLoadConfig(MODEL_PATH, sid_dims=(32, 32, 32)),
                          device="cuda")
m_check.eval()

with torch.no_grad():
    seq_logp, out, info = sequence_logp(
        m_check, tok, sid, prompts, chosens, max_length=2048,
    )
    labels = info["labels"]
    prefix_len = out.logits.size(1) - labels.size(1)
    pad = torch.full((labels.size(0), prefix_len), -100,
                     dtype=labels.dtype, device=labels.device)
    full = torch.cat([pad, labels], dim=1)
    n_valid = full[:, 1:].ne(-100).sum().item()
    manual = -seq_logp.sum().item() / n_valid
    print(f"  total valid tokens (chosen target tokens summed across batch): {n_valid}")
    print(f"  manual avg NLL (from sequence_logp):  {manual:.4f}")
    print(f"  HF base_model.out.loss (avg NLL)   :  {out.loss.item():.4f}")
    delta = abs(manual - out.loss.item())
    print(f"  diff:  {delta:.6f}    {'✅ 一致' if delta < 1e-3 else '❌ 不一致, sequence_logp 有 bug'}")

del m_check
torch.cuda.empty_cache()

# ============================================================ [B] 对比 ====
banner("[B] pi_chosen / pi_rejected: 'random init' vs 'DPO trained'")

m_random = build_sid_model(SidModelLoadConfig(MODEL_PATH, sid_dims=(32, 32, 32)),
                           device="cuda")
m_random.eval()

m_dpo = build_sid_model(SidModelLoadConfig(MODEL_PATH, sid_dims=(32, 32, 32)),
                        device="cuda")
m_dpo.sid_prefix.load_state_dict(
    torch.load(DPO_CKPT, map_location="cuda")["sid_prefix"], strict=True,
)
m_dpo.eval()

with torch.no_grad():
    pi_c_r, _, _ = sequence_logp(m_random, tok, sid, prompts, chosens,   2048)
    pi_r_r, _, _ = sequence_logp(m_random, tok, sid, prompts, rejecteds, 2048)
    pi_c_d, _, _ = sequence_logp(m_dpo,    tok, sid, prompts, chosens,   2048)
    pi_r_d, _, _ = sequence_logp(m_dpo,    tok, sid, prompts, rejecteds, 2048)

print(f"  {'sample':<6s}  {'item_id':<14s}  {'sid':<14s}  "
      f"{'rand: π_c-π_r':>14s}  {'dpo: π_c-π_r':>14s}  Δ(dpo-rand)")
for i in range(len(rows)):
    rand_diff = (pi_c_r[i] - pi_r_r[i]).item()
    dpo_diff  = (pi_c_d[i] - pi_r_d[i]).item()
    delta     = dpo_diff - rand_diff
    print(f"  #{i+1:<5d}  {rows[i]['item_id']:<14s}  {str(rows[i]['sid']):<14s}  "
          f"{rand_diff:>14.4f}  {dpo_diff:>14.4f}  {delta:+.4f}")

print()
print("  解读:")
print("    - 任何一行 dpo Δ > rand Δ  → DPO 在该样本上把 chosen 相对于 rejected 推高了")
print("    - 平均 Δ(dpo-rand) > 0      → DPO 总体方向正确, 只是幅度小")
print("    - 平均 Δ(dpo-rand) ≈ 0      → DPO 没学到东西 (噪声游走)")
print("    - 平均 Δ(dpo-rand) < 0      → DPO 反向跑了, 训练真的有问题")

# ============================================================ [C] 权重 ====
banner("[C] DPO ckpt vs fresh random prefix (实际 prefix 推动量)")

m_fresh = build_sid_model(SidModelLoadConfig(MODEL_PATH, sid_dims=(32, 32, 32)),
                          device="cpu")
fresh_state = m_fresh.sid_prefix.state_dict()
del m_fresh
dpo_state = torch.load(DPO_CKPT, map_location="cpu")["sid_prefix"]

print(f"  {'param':<35s}  {'shape':<22s}  {'|fresh|':>10s}  {'|fresh-dpo|':>12s}  rel")
total_d, total_n = 0.0, 0.0
for k in fresh_state:
    if k not in dpo_state:
        continue
    f = fresh_state[k].float()
    d = dpo_state[k].float()
    fnorm = f.abs().mean().item()
    diff  = (f - d).abs().mean().item()
    rel   = diff / (fnorm + 1e-9)
    total_d += (f - d).abs().sum().item()
    total_n += f.abs().sum().item()
    print(f"  {k:<35s}  {str(tuple(f.shape)):<22s}  "
          f"{fnorm:>10.4f}  {diff:>12.4f}  {rel:>6.2%}")

print(f"\n  Overall relative change vs fresh random init: {total_d/total_n:.2%}")
print()
print("  解读:")
print("    - 各张量 rel 大致 5-30%   → DPO 真的训过, 推动幅度合理")
print("    - 各张量 rel < 1%         → DPO 几乎没动 (梯度噪声互相抵消)")
print("    - 某张量 rel > 100%       → 训飞了, prefix 走出了正常区间")

# ============================================================ [D] EOS ====
banner("[D] 模型在'标题写完应该 EOS'的位置, 给 EOS 多大概率")

# 构造 "已经写完一个标题, 接下来该停" 的输入: prompt + "\nTitle: <chosen>"
# 看模型对下一 token 的 top-K, 重点看 <|im_end|> (248046) 排第几

EOS_IM_END     = 248046   # <|im_end|> = tokenizer.eos_token_id, 训练拼的就是它
EOS_ENDOFTEXT  = 248044   # <|endoftext|> = base config 的 EOS, 预训练用的

@torch.no_grad()
def probe_eos_at_title_end(model, prompt, chosen_title, sid_t):
    """让模型看到 'prompt + \\nTitle: <chosen>', 看下一步它最想吐什么"""
    text = f"{prompt}\nTitle: {chosen_title}"
    inp = tok([text], return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    out = model(input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                sid=sid_t)
    next_logits = out.logits[:, -1, :]
    probs = next_logits.softmax(-1)
    top10 = torch.topk(next_logits, 10, dim=-1)
    return {
        "top10": [(top10.indices[0, i].item(),
                   top10.values[0, i].item(),
                   probs[0, top10.indices[0, i]].item(),
                   tok.decode([top10.indices[0, i].item()]))
                  for i in range(10)],
        "p_im_end":     probs[0, EOS_IM_END].item(),
        "p_endoftext":  probs[0, EOS_ENDOFTEXT].item(),
        "rank_im_end":  (next_logits[0] >= next_logits[0, EOS_IM_END]).sum().item(),
        "rank_endoftext": (next_logits[0] >= next_logits[0, EOS_ENDOFTEXT]).sum().item(),
    }

print(f"  比较两个模型对 <|im_end|> 的态度:")
print(f"    - rank=1 → 模型最想吐 EOS, 推理一定会停")
print(f"    - rank=高 + p 极小 → 模型不想吐 EOS, 推理会撞 max_new_tokens")
print()

for i, row in enumerate(rows[:2]):
    sid_one = sid[i:i+1].cuda()
    print(f"  --- Sample {i+1}: item={row['item_id']} sid={row['sid']} ---")
    print(f"    chosen title: {row['title_chosen']!r}")
    print()

    res_r = probe_eos_at_title_end(m_random, prompts[i], row["title_chosen"], sid_one)
    res_d = probe_eos_at_title_end(m_dpo,    prompts[i], row["title_chosen"], sid_one)

    print(f"    [random init prefix]  p(<|im_end|>)={res_r['p_im_end']:.2e}  "
          f"rank={res_r['rank_im_end']}    "
          f"p(<|endoftext|>)={res_r['p_endoftext']:.2e}  rank={res_r['rank_endoftext']}")
    print(f"    [DPO trained prefix]  p(<|im_end|>)={res_d['p_im_end']:.2e}  "
          f"rank={res_d['rank_im_end']}    "
          f"p(<|endoftext|>)={res_d['p_endoftext']:.2e}  rank={res_d['rank_endoftext']}")
    print()
    print(f"    [DPO] top-10 candidates after title (这里就是模型'最想接什么'):")
    for tid, lg, p, dec in res_d["top10"]:
        flag = ""
        if tid == EOS_IM_END:    flag = "  ← <|im_end|>  (我们设的 stop token)"
        if tid == EOS_ENDOFTEXT: flag = "  ← <|endoftext|>  (base config 的 EOS)"
        print(f"      id={tid:>7d}  logit={lg:7.3f}  p={p:.4f}  {dec!r}{flag}")
    print()

print("  解读:")
print("    - DPO 那行 p(<|im_end|>) 仍然 < 1e-3, rank 几十名开外")
print("      → 模型根本不想吐它, 跟推理 EOS 配置无关, 是模型没学会")
print("    - rank 越靠前但仍没采到 → 是 LR/dropout 噪声问题")
print("    - rank=1 但推理还是不停 → 推理代码 break 逻辑真出 bug 了 (这种情况别先做下面)")

print()
print("=" * 72)
print("  Done. 把 [A] [B] [C] [D] 四段输出整段贴回来, 我能定位问题:")
print("    - [A] 失败 → sequence_logp / forward 有 bug")
print("    - [B] Δ ≈ 0 → DPO 信号本身是噪声 (本质问题, 不是代码)")
print("    - [B] Δ < 0 → DPO 反向更新 (代码层面有 bug)")
print("    - [C] 几乎不动 → 梯度被噪声抵消, 跟 [B] 互相印证")
print("    - [D] EOS rank 靠后 → 不是配置 bug, 模型根本没学会吐 EOS (你需要 SFT)")
print("    - [D] EOS rank=1 但还是不停 → wrapper.generate break 逻辑 bug (回头查)")
print("=" * 72)
