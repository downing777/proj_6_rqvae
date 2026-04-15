# SID 标题训练数据格式

本目录只定义格式，不做真实样本构建。

## 文件
- sft_samples.schema.json: SFT 数据 schema
- dpo_pairs.schema.json: DPO 数据 schema
- sft_samples.example.jsonl: SFT 示例数据
- dpo_pairs.example.jsonl: DPO 示例数据

## 约束建议
- sid 使用固定三元组: [rqid_0, rqid_1, rqid_2]
- DPO 负样本建议混合:
  - hard_cross_sid: 70%
  - easy_random: 30%
