# SID Softprompt Pipeline

本目录实现了 SID 驱动的个性化标题训练与推理流程：
- SID 条件前缀（Hyper Prefix）
- SFT 训练
- DPO 训练（含 reference model）
- 离线评估与回退策略

## 目录结构
- `models/sid_prefix.py`: SID 三元组 -> 多虚拟 token 前缀
- `models/wrapper.py`: 将 SID 前缀注入 CausalLM
- `data/build_dpo_pairs.py`: 只产出正负样本格式 schema + 示例
- `train/train_sft.py`: SFT 训练入口
- `train/train_dpo.py`: DPO 训练入口
- `infer/generate_title.py`: 按 `(item, sid)` 生成标题
- `eval/offline_eval.py`: 偏好胜率、关键词召回、Distinct-n、回退比例

## 1) 写出样本格式（不构造真实数据）
```bash
python softprompt/data/build_dpo_pairs.py \
  --output-dir softprompt/data/format
```

## 2) SFT
```bash
python softprompt/train/train_sft.py \
  --train-jsonl softprompt/data/format/sft_samples.example.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --sid-dims 32,32,32 \
  --freeze-backbone \
  --output-dir softprompt/outputs/sft
```

## 3) DPO
```bash
python softprompt/train/train_dpo.py \
  --train-jsonl softprompt/data/format/dpo_pairs.example.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --sft-ckpt softprompt/outputs/sft/sid_sft.pt \
  --sid-dims 32,32,32 \
  --freeze-backbone \
  --beta 0.1 \
  --output-dir softprompt/outputs/dpo
```

## 4) 推理
输入 `jsonl` 每行格式：
```json
{"item_id":"B0-123","sid":[1,2,3],"context":"品牌: XX; 品类: 蓝牙耳机; 卖点: 长续航, 降噪, 舒适佩戴"}
```

运行：
```bash
python softprompt/infer/generate_title.py \
  --input-jsonl softprompt/data/format/sft_samples.example.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --sid-ckpt softprompt/outputs/dpo/sid_dpo.pt \
  --num-beams 1 \
  --output-jsonl softprompt/outputs/predictions.jsonl
```

注意：当前 wrapper 为了兼容本地 `transformers` 版本，推理仅支持 `--num-beams 1`。

## 5) 离线评估 + 回退门控
```bash
python softprompt/eval/offline_eval.py \
  --dpo-jsonl softprompt/data/format/dpo_pairs.example.jsonl \
  --pred-jsonl softprompt/outputs/predictions.jsonl \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --sid-ckpt softprompt/outputs/dpo/sid_dpo.pt \
  --output-json softprompt/outputs/offline_eval.json
```
