# RQ-VAE 训练详解

本文档详细解释 `rqvae/` 中 RQ-VAE 的模型结构、训练流程、数据集构建方式以及 Batch 的采样策略。所有分析基于 `rqvae/train_rqvae.py`（物品 Semantic ID 训练）。

---

## 目录

1. [整体架构概览](#整体架构概览)
2. [模型结构](#模型结构)
   - [Encoder / Decoder (MLP)](#encoder--decoder-mlp)
   - [单层向量量化 (Quantize)](#单层向量量化-quantize)
   - [残差量化主流程 (RqVae)](#残差量化主流程-rqvae)
3. [损失函数](#损失函数)
4. [数据集：ItemData](#数据集-itemdata)
5. [Batch 的采样方式](#batch-的采样方式)
6. [完整训练循环](#完整训练循环)
7. [评估：Codebook 质量度量](#评估-codebook-质量度量)
8. [SemanticIdTokenizer：预计算语料库 ID](#semanticidtokenizer预计算语料库-id)
9. [端到端数据流示意](#端到端数据流示意)

---

## 整体架构概览

RQ-VAE（Residual Quantization Variational Autoencoder）在本项目中的作用是将每个**物品**的连续嵌入向量压缩为一个**离散的多层 Codebook 索引元组**，作为推荐系统的 Semantic ID：

```
物品描述文本  +  分类特征（流派等）
         │
         ▼  Sentence-T5-XXL / 人工特征
  [N_items, 768]  连续文本嵌入
         │
         ▼  RQ-VAE 训练（train_rqvae.py）
  [N_items, 3]   离散 Semantic ID
                 每个物品 = (id_0, id_1, id_2)
                 ├── id_0：主方向粗粒度 ID  (0~codebook_size-1)
                 ├── id_1：一阶残差细粒度 ID
                 └── id_2：二阶残差细粒度 ID
```

这个 3 元组 ID 被后续的推荐 Transformer 当作词元输入，用前缀树做高效的层次化候选检索。

---

## 模型结构

### Encoder / Decoder (MLP)

**文件**：`rqvae/modules/encoder.py`

Encoder 和 Decoder 共用同一个 `MLP` 类，结构是多层全连接 + SiLU 激活：

```
Encoder: [input_dim] → Linear → SiLU → ... → [embed_dim]   (可选 L2 归一化)
Decoder: [embed_dim] → Linear → SiLU → ... → [input_dim]   (强制 L2 归一化)
```

关键设计：
- 所有 Linear 层**不带 bias**（`bias=False`）
- 激活函数为 **SiLU**（Sigmoid Linear Unit），比 ReLU 更平滑
- Decoder 输出强制做 **L2 归一化**，使重构向量落在单位球上，与文本嵌入的 cosine 相似性空间对齐

### 单层向量量化 (Quantize)

**文件**：`rqvae/modules/quantize.py`

每一层 VQ 执行以下操作：

#### 1. Codebook 的延迟 K-Means 初始化

```python
# rqvae/init/kmeans.py
def kmeans_init_(tensor, x):
    kmeans_out = Kmeans(k=codebook_size).run(x)
    tensor.data.copy_(kmeans_out.centroids)
```

**第一次 forward 时**（且 `codebook_kmeans_init=True`），用当前输入 `x` 跑纯 PyTorch 实现的 Lloyd K-Means 算法，将 codebook 向量初始化为数据分布中真实的聚类中心。相比随机初始化，K-Means 初始化能：
- 避免 codebook 塌缩（大量 codebook 向量从未被使用）
- 显著加速早期收敛

在 `train_rqvae.py` 中，第 0 步专门用前 **20,000 条物品**触发 K-Means 初始化（见[完整训练循环](#完整训练循环)）。

#### 2. 最近邻查找（硬分配）

```python
dist = ||x||² + ||c||² - 2 * x @ C.T   # L2 距离平方，广播计算
ids  = argmin(dist)                      # 硬分配，不参与梯度
```

#### 3. 可微分梯度估计：三种模式

代码支持三种量化梯度估计方式，通过 `codebook_mode` 参数控制：

| 模式 | 机制 | 特点 |
|---|---|---|
| **`GUMBEL_SOFTMAX`** | 用 Gumbel 噪声对 codebook 做 soft 加权求和 | 默认模式，引入随机性 |
| **`STE`** | `x + (emb - x).detach()`，梯度直通 | 有偏，忽略量化跳跃 |
| **`ROTATION_TRICK`** | 用 Householder 变换将 encoder 输出旋转至最近 codebook 向量 | 无偏，梯度最准确（arXiv:2410.06424） |

**Rotation Trick 的直觉**：将 encoder 输出 `z` 的**方向**旋转到与选中的 codebook 向量 `e` 方向一致，同时按照 `||e|| / ||z||` 缩放模长，从而既保持离散选择，又允许准确的梯度流回 encoder。

#### 4. 量化损失（QuantizeLoss）

```python
emb_loss   = ||z.detach() - e||²   # 推动 codebook 向量靠近 encoder 输出（EMA 替代方案）
query_loss = ||z - e.detach()||²   # commitment loss：推动 encoder 输出靠近 codebook
loss = emb_loss + commitment_weight * query_loss
```

`commitment_weight=0.25` 控制 encoder 被拉向 codebook 的力度，防止 encoder 发散太快。

### 残差量化主流程 (RqVae)

**文件**：`rqvae/modules/rqvae.py`

`get_semantic_ids` 实现了残差量化的核心循环：

```python
z = encoder(x)      # [N, embed_dim]
residual = z
sem_ids, embs = [], []

for layer in self.layers:              # n_layers 层（通常 3 层）
    out = layer(residual)              # 找最近 codebook 向量
    residual = residual - out.emb      # 计算残差，传给下一层
    sem_ids.append(out.ids)
    embs.append(out.emb)

x_hat = decoder(sum(embs))            # 所有层 codebook 向量求和后解码
```

**残差量化的直觉**：
- 第 0 层用 codebook 捕捉 `z` 的主方向 → `emb_0` 接近 `z`，残差 `z - emb_0` 较小
- 第 1 层对这个小残差再做量化 → `emb_1` 捕捉第 0 层的误差
- 第 2 层继续细化……
- 最终 `emb_0 + emb_1 + emb_2` 近似还原 `z`，而三个整数索引 `(id_0, id_1, id_2)` 就是该物品的 Semantic ID

---

## 损失函数

**文件**：`rqvae/modules/loss.py`

```
total_loss = reconstruction_loss + quantize_loss
```

### 重构损失（Reconstruction Loss）

```python
reconstruction_loss = ((x_hat - x) ** 2).sum(axis=-1).mean()
```

**MSE（sum over feature dim）**。`x_hat` 是 decoder 输出，`x` 是原始物品嵌入（文本维度 768）。

重构目标是原始输入 `x` 而非量化后的 `z`，这迫使模型在量化过程中保留完整的语义信息。

### CategoricalReconstructionLoss（含分类特征时）

```python
# 连续维度（文本嵌入）：MSE
reconstr = ((x_hat[:, :-n_cat] - x[:, :-n_cat]) ** 2).sum(axis=-1)
# 分类维度（流派 one-hot）：BCE
cat_reconstr = binary_cross_entropy_with_logits(
    x_hat[:, -n_cat:], x[:, -n_cat:], reduction='none'
).sum(axis=-1)
total = reconstr + cat_reconstr
```

当物品特征包含类别维度（如电影流派的 18 维 one-hot），文本部分用 MSE，类别部分用 BCE，两者相加。`n_cat_feats` 在 gin config 中配置（Amazon/ML-1M 数据集典型值为 18）。

### 量化损失（逐层累加）

```
quantize_loss = Σ_{i=0}^{n_layers-1} (emb_loss_i + 0.25 * query_loss_i)
```

---

## 数据集：ItemData

**文件**：`rqvae/data/processed.py`

```python
class ItemData(Dataset):
    def __init__(self, root, dataset, train_test_split="all", split="beauty"):
        raw_data = AmazonReviews(root, split=split)   # 或 MovieLens
        raw_data.process(max_seq_len=20)              # 如果未处理则触发预处理

        if train_test_split == "train":
            filt = raw_data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_data.data["item"]["is_train"]
        else:
            filt = all_true

        self.item_data = raw_data.data["item"]["x"][filt]  # [N, 768+]

    def __getitem__(self, idx):
        x = self.item_data[idx, :768]                 # 只取文本嵌入维度
        return SeqBatch(
            user_ids = -1,
            ids      = idx,
            ids_fut  = -1,
            x        = x,          # 真正有意义的输入, rq_vae 阶段只用到了这个
            x_fut    = -1,
            seq_mask = True
        )
```

关键点：
- **物品特征**来自预处理好的 `.pt` 文件，第 0~767 维是 Sentence-T5-XXL 文本嵌入，之后是类别特征
- `__getitem__` 默认只取 `:768`（文本维度），类别特征由 `CategoricalReconstructionLoss` 内部按 `n_cat_feats` 分割处理
- `SeqBatch` 是统一的数据格式，`user_ids`、`ids_fut`、`x_fut` 在 Item 训练中均为占位符（`-1`），只有 `x` 有效

**数据分割**：训练时用 `"train"` 分割，评估时用 `"eval"` 分割，计算 corpus ID 时用 `"all"` 分割（全量物品）。

---

## Batch 的采样方式

**文件**：`rqvae/train_rqvae.py`

### 训练集采样

```python
train_dataset = ItemData(root=dataset_folder, dataset=dataset,
                         train_test_split="train", split=dataset_split)

train_sampler = BatchSampler(
    RandomSampler(train_dataset),   # 随机打乱索引
    batch_size,                     # 每个 batch 的物品数
    drop_last=False                 # 保留最后一个不满的 batch
)

train_dataloader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=None,               # 由 BatchSampler 控制 batch 大小，这里设 None
    collate_fn=lambda batch: batch # batch 已经是 SeqBatch，直接透传
)

train_dataloader = cycle(train_dataloader)  # 无限循环生成器
```

**`cycle` 的实现**（`rqvae/data/utils.py`）：

```python
def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data
```

`cycle` 将 DataLoader 包装成一个**无限生成器**：数据集遍历完后自动从头开始，不需要手动重置迭代器。配合固定 step 数（`iterations=50000`）训练，等效于多个 epoch 的随机采样。

**取 batch 的方式**（`rqvae/data/utils.py`）：

```python
def next_batch(dataloader, device):
    batch = next(dataloader)                         # 从无限生成器取一个 batch
    return SeqBatch(*[v.to(device) for v in batch]) # 移到训练设备
```

### 评估集采样

评估时不用 `cycle`，而是顺序遍历整个 eval 集：

```python
eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, drop_last=False)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, ...)
for batch in eval_dataloader:
    ...   # 遍历一遍 eval 集
```

### 与 train_user.py 的对比

| 维度 | `train_rqvae.py`（原始） | `train_user.py`（用户版） |
|---|---|---|
| 采样器 | `BatchSampler(RandomSampler(...))` | `DataLoader(shuffle=True)` |
| 无限循环 | `cycle()` 包装，永远不 StopIteration | `try/except StopIteration` 手动重置 |
| 取 batch | `next_batch(dataloader, device)` | `next(it)[0].to(device)` |
| Accelerate | 是（支持多卡 / AMP） | 否（纯单卡） |

---

## 完整训练循环

**文件**：`rqvae/train_rqvae.py`

```python
accelerator = Accelerator(split_batches=True, mixed_precision="fp16" if amp else "no")

model = RqVae(
    input_dim        = vae_input_dim,        # gin config 中指定
    embed_dim        = vae_embed_dim,
    hidden_dims      = vae_hidden_dims,
    codebook_size    = vae_codebook_size,
    codebook_kmeans_init = use_kmeans_init,  # True
    codebook_mode    = vae_codebook_mode,    # GUMBEL_SOFTMAX（默认）
    n_layers         = vae_n_layers,         # 3
    n_cat_features   = vae_n_cat_feats,      # 18（有分类特征时）
    commitment_weight= commitment_weight,    # 0.25
)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Accelerate 接管模型、优化器、DataLoader（支持 DDP / AMP）
model, optimizer = accelerator.prepare(model, optimizer)
train_dataloader  = accelerator.prepare(train_dataloader)

t = 0.2   # Gumbel 温度，全程固定

for iter in range(iterations):   # 默认 50000 步

    model.train()

    # ── Step 0：用前 20000 条物品触发 K-Means 初始化 ──────────────
    if iter == 0 and use_kmeans_init:
        kmeans_init_data = batch_to(
            train_dataset[torch.arange(min(20000, len(train_dataset)))],
            device
        )
        model(kmeans_init_data, t)   # 触发每层 Quantize 的 K-Means 初始化

    # ── 梯度清零 ──────────────────────────────────────────────────
    optimizer.zero_grad()

    # ── 梯度累积（gradient_accumulate_every 次 forward） ───────────
    total_loss = 0
    for _ in range(gradient_accumulate_every):
        data = next_batch(train_dataloader, device)

        with accelerator.autocast():                  # 可选 fp16 AMP
            model_output = model(data, gumbel_t=t)
            #  内部流程：
            #  1. encoder(x)  →  z
            #  2. Layer 0 Quantize(z)     → id_0, emb_0, residual = z - emb_0
            #  3. Layer 1 Quantize(res)   → id_1, emb_1, residual = res - emb_1
            #  4. Layer 2 Quantize(res)   → id_2, emb_2
            #  5. decoder(emb_0+emb_1+emb_2) → x_hat
            #  6. loss = ReconstructionLoss(x_hat, x) + Σ QuantizeLoss_i
            loss = model_output.loss / gradient_accumulate_every
            total_loss += loss

    # ── 反向传播 + 参数更新 ───────────────────────────────────────
    accelerator.backward(total_loss)     # 兼容 DDP 的 backward
    accelerator.wait_for_everyone()
    optimizer.step()
    accelerator.wait_for_everyone()

    # ── 日志（wandb）────────────────────────────────────────────
    if wandb_logging and accelerator.is_main_process:
        wandb.log({
            "total_loss":          total_loss,
            "reconstruction_loss": model_output.reconstruction_loss,
            "rqvae_loss":          model_output.rqvae_loss,
            "temperature":         t,
            "p_unique_ids":        model_output.p_unique_ids,  # batch 内 unique ID 比例
            "emb_avg_norm_0/1/2":  ...,                        # 各层嵌入模长均值
        })

    # ── 定期评估（eval_every 步）────────────────────────────────
    if (iter+1) % eval_every == 0:
        _run_eval(model, eval_dataloader, ...)
        _run_id_diversity(model, tokenizer, index_dataset, ...)

    # ── 定期保存 checkpoint ───────────────────────────────────────
    if (iter+1) % save_model_every == 0:
        torch.save({"iter": iter, "model": ..., "optimizer": ...}, path)
```

### 关于 Gumbel 温度

训练时 `t=0.2` **全程固定**，没有退火调度。推理时 `SemanticIdTokenizer` 调用 `get_semantic_ids` 时温度极低（接近 0），退化为 hard argmax，得到确定性的离散 ID。

### 关于 Accelerate

- `split_batches=True`：多卡时每张卡拿到 `batch_size/n_gpus` 个样本（而非每张卡各拿 `batch_size`）
- `mixed_precision="fp16"`：启用 AMP 时用 fp16 做 forward/backward，参数更新仍用 fp32
- `accelerator.backward(loss)`：在 DDP 模式下自动处理梯度同步

---

## 评估：Codebook 质量度量

**每 `eval_every` 步**，在 main process 上执行以下评估：

### 1. Eval 集重构损失

```python
model.eval()
for batch in eval_dataloader:
    with torch.no_grad():
        out = model(batch, gumbel_t=t)
    # 收集 out.loss / out.reconstruction_loss / out.rqvae_loss
eval_losses = mean(eval_losses)
```

### 2. Codebook 利用率（Codebook Usage）

```python
corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)  # [N_items, n_layers+1]

for cid in range(n_layers):
    unique_ids = len(torch.unique(corpus_ids[:, cid]))
    codebook_usage = unique_ids / codebook_size   # 越接近 1.0 越好
```

每层 codebook 中被实际使用的向量比例。若利用率低（如 < 0.3），说明大量 codebook 向量从未被选中（codebook 塌缩）。

### 3. Semantic ID 分布熵（RQ-VAE Entropy）

```python
_, counts = torch.unique(corpus_ids[:, :-1], dim=0, return_counts=True)  # 排除去重列
p = counts / N_items
rqvae_entropy = -(p * torch.log(p)).sum()
```

所有物品的 `(id_0, id_1, id_2)` 三元组的分布熵。熵越高，说明 ID 分布越均匀，物品之间区分度越好。

### 4. 最大 ID 重复数（Max ID Duplicates）

```python
max_duplicates = corpus_ids[:, -1].max() / N_items
```

`corpus_ids` 的最后一列记录了每个物品的 ID 在整个语料库中出现了多少次（由 `precompute_corpus_ids` 计算）。`max_duplicates` 反映最坏情况下的 ID 碰撞程度。理想情况下每个物品有唯一 ID，此值趋向 `1/N_items`。

---

## SemanticIdTokenizer：预计算语料库 ID

**文件**：`rqvae/modules/tokenizer/semids.py`

`SemanticIdTokenizer` 是 RQ-VAE 训练完成后的推理接口，负责：
1. 对全量物品预计算 Semantic ID（`precompute_corpus_ids`）
2. 在推荐 Transformer 训练时将物品历史序列 tokenize 成 Semantic ID 序列

### precompute_corpus_ids

```python
@torch.no_grad()
def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
    sampler = BatchSampler(SequentialSampler(range(len(movie_dataset))),
                           batch_size=512, drop_last=False)
    dataloader = DataLoader(movie_dataset, sampler=sampler, shuffle=False, ...)

    cached_ids = None
    dedup_dim = []

    for batch in dataloader:
        batch_ids = self.forward(batch).sem_ids      # [512, n_layers]

        # 检测 batch 内部重复
        is_hit = self._get_hits(batch_ids, batch_ids)
        hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)

        # 检测与已缓存 ID 的重复
        if cached_ids is not None:
            is_hit = self._get_hits(batch_ids, cached_ids)
            hits += is_hit.sum(axis=-1)
            cached_ids = pack([cached_ids, batch_ids], "* d")[0]
        else:
            cached_ids = batch_ids.clone()

        dedup_dim.append(hits)

    # 最终 cached_ids: [N_items, n_layers+1]，最后一列是重复计数
    self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]
    return self.cached_ids
```

预计算完成后，`self.cached_ids[item_id]` 就是该物品的 Semantic ID（+重复计数），后续 Transformer 训练时直接查表，不需要再过 RQ-VAE。

---

## 端到端数据流示意

```
原始数据集（Amazon/MovieLens）
       │
       ├─ 预处理（PreprocessingMixin）
       │    ├─ 物品描述文本 → Sentence-T5-XXL → [N, 768] 文本嵌入
       │    ├─ 物品分类特征 → one-hot → [N, 18]
       │    ├─ 用户交互序列 → 时间排序 → rolling window
       │    └─ 存为 data.pt（后续直接加载）
       │
       ├─ ItemData（train split）
       │    └─ __getitem__(idx) → SeqBatch(x=item_emb[:768])
       │
       ├─ BatchSampler(RandomSampler) → cycle()
       │    └─ 无限随机抽取 batch_size 个物品
       │
       ├─ 训练循环（50000 步）
       │    ├─ iter=0: 前 20000 条物品做 K-Means 初始化 codebook
       │    └─ 每步：
       │         batch ← next_batch(dataloader, device)
       │         z    ← MLP encoder(batch.x)
       │         Layer 0: Quantize(z)       → id_0, emb_0
       │                  residual_1 = z - emb_0
       │         Layer 1: Quantize(res_1)   → id_1, emb_1
       │                  residual_2 = res_1 - emb_1
       │         Layer 2: Quantize(res_2)   → id_2, emb_2
       │         x_hat ← MLP decoder(emb_0 + emb_1 + emb_2)
       │         loss  = ReconstructionLoss(x_hat, x)
       │                 + Σ QuantizeLoss_i(commitment + emb_push)
       │         accelerator.backward(loss) → optimizer.step()
       │
       ├─ 每 eval_every 步：
       │    ├─ eval split 上计算重构损失
       │    └─ precompute_corpus_ids(all items)
       │         ├─ codebook_usage（各层利用率）
       │         ├─ rqvae_entropy（ID 分布熵）
       │         └─ max_id_duplicates（最大 ID 碰撞率）
       │
       └─ 训练结束：保存 checkpoint
            ├─ model.state_dict()
            ├─ optimizer.state_dict()
            └─ iter（支持断点续训）
```
