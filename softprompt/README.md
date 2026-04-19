# SID Softprompt Pipeline

本目录实现了 SID 驱动的个性化标题训练与推理流程：
- SID 条件前缀（Hyper Prefix）
- SFT 训练
- DPO 训练（含 reference model）
- 离线评估与回退策略

默认以 `Qwen/Qwen3-0.6B` 跑通链路，标题为中文风格。

## 目录结构
- `models/sid_prefix.py`: SID 三元组 -> 多虚拟 token 前缀
- `models/wrapper.py`: 将 SID 前缀注入 CausalLM
- `data/build_dpo_pairs.py`: 只产出 schema 与最小示例
- `data/make_mock_sid_caption_data.py`: 伪造 SID + SFT/DPO 训练数据
- `train/train_sft.py`: SFT 训练入口
- `train/train_dpo.py`: DPO 训练入口
- `infer/generate_title.py`: 按 `(item, sid)` 生成标题
- `eval/offline_eval.py`: 偏好胜率、关键词召回、Distinct-n、回退比例

## 0) 环境准备
建议 Python `3.10+`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r softprompt/requirements.txt
```

## 1) 造一批伪数据（美国 item 场景）
```bash
python softprompt/data/make_mock_sid_caption_data.py \
  --output-dir softprompt/data/mock \
  --sid-dims 32,32,32 \
  --num-items 6 \
  --sids-per-item 4 \
  --hard-negative-ratio 0.7 \
  --seed 42
```

输出：
- `softprompt/data/mock/sft_train.jsonl`
- `softprompt/data/mock/dpo_train.jsonl`

## 2) SFT
```bash
python softprompt/train/train_sft.py \
  --train-jsonl softprompt/data/mock/sft_train.jsonl \
  --base-model Qwen/Qwen3-0.6B \
  --sid-dims 32,32,32 \
  --freeze-backbone \
  --train-batch-size 4 \
  --epochs 1 \
  --output-dir softprompt/outputs/sft
```

## 3) DPO
```bash
python softprompt/train/train_dpo.py \
  --train-jsonl softprompt/data/mock/dpo_train.jsonl \
  --base-model Qwen/Qwen3-0.6B \
  --sft-ckpt softprompt/outputs/sft/sid_sft.pt \
  --sid-dims 32,32,32 \
  --freeze-backbone \
  --beta 0.1 \
  --train-batch-size 2 \
  --epochs 1 \
  --output-dir softprompt/outputs/dpo
```

## 4) 推理生成标题
输入 `jsonl` 每行格式：
```json
{"item_id":"US-ELEC-001","sid":[1,2,3],"context":"站点: 美国; 品牌: SoundPeak; 品类: 蓝牙耳机; 核心卖点: 主动降噪, 36小时续航, 通话清晰; 人群偏好: 功能党, 参数够硬"}
```

运行：
```bash
python softprompt/infer/generate_title.py \
  --input-jsonl softprompt/data/mock/sft_train.jsonl \
  --base-model Qwen/Qwen3-0.6B \
  --sid-ckpt softprompt/outputs/dpo/sid_dpo.pt \
  --num-beams 1 \
  --output-jsonl softprompt/outputs/predictions.jsonl
```

注意：当前 wrapper 仅支持 `--num-beams 1`。

## 5) 离线评估 + 回退门控
```bash
python softprompt/eval/offline_eval.py \
  --dpo-jsonl softprompt/data/mock/dpo_train.jsonl \
  --pred-jsonl softprompt/outputs/predictions.jsonl \
  --base-model Qwen/Qwen3-0.6B \
  --sid-ckpt softprompt/outputs/dpo/sid_dpo.pt \
  --output-json softprompt/outputs/offline_eval.json
```

## 6) 一键跑通（最小烟雾版）
如果只是先确认链路通，建议缩小样本：

```bash
python softprompt/data/make_mock_sid_caption_data.py --output-dir softprompt/data/mock_smoke --num-items 3 --sids-per-item 2
python softprompt/train/train_sft.py --train-jsonl softprompt/data/mock_smoke/sft_train.jsonl --base-model Qwen/Qwen3-0.6B --sid-dims 32,32,32 --freeze-backbone --epochs 1 --train-batch-size 2 --output-dir softprompt/outputs/sft_smoke
python softprompt/train/train_dpo.py --train-jsonl softprompt/data/mock_smoke/dpo_train.jsonl --base-model Qwen/Qwen3-0.6B --sft-ckpt softprompt/outputs/sft_smoke/sid_sft.pt --sid-dims 32,32,32 --freeze-backbone --epochs 1 --train-batch-size 1 --output-dir softprompt/outputs/dpo_smoke
python softprompt/infer/generate_title.py --input-jsonl softprompt/data/mock_smoke/sft_train.jsonl --base-model Qwen/Qwen3-0.6B --sid-ckpt softprompt/outputs/dpo_smoke/sid_dpo.pt --num-beams 1 --output-jsonl softprompt/outputs/predictions_smoke.jsonl
```
