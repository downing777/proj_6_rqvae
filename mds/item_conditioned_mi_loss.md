# Item-Conditioned MI 正则实现说明（`train_user.py`）

本文详细解释当前在 `train_user.py` 中实现的 item-conditioned clustering loss（互信息风格正则），用于实现：

- 同一 item 下的 user semantic id 更集中（降低局部熵）
- 不同 item 之间整体用码更分散（提高全局熵）

---

## 1. 设计目标与核心思想

我们希望在不破坏现有 RQ-VAE 主体结构（重建 + VQ/commitment）的前提下，加入一个可微的聚类约束。

目标函数（每层）：

$$
\mathcal{L}^{(k)}_{reg} = \alpha \cdot H(C^{(k)}|I) - \beta \cdot H(C^{(k)})
$$

总正则：

$$
\mathcal{L}_{reg}=\sum_{k=1}^{L}\mathcal{L}^{(k)}_{reg}
$$

最终训练损失：

$$
\mathcal{L}_{total}=\mathcal{L}_{base}+\lambda\cdot\mathcal{L}_{reg}
$$

其中：

- $\mathcal{L}_{base}$：原有 RQ-VAE 损失（reconstruction + quantize）
- $\lambda$：`mi_weight`
- $L$：用于正则的层数（`mi_reg_layers`）

---

## 2. 为什么不改 `quantize.py` 里的 `QuantizeLoss`

当前实现**不需要**改 `rqvae/modules/quantize.py` 的 `QuantizeLoss`，原因：

1. `QuantizeLoss` 负责向量量化本身（codebook 对齐与 commitment），是基础训练目标。
2. 你的 item 聚类需求是“额外结构约束”，更适合作为附加正则项，而不是替换 VQ loss。
3. 在 `train_user.py` 叠加正则可以：
   - 独立开关（`--enable-item-mi-loss`）
   - 不影响原有训练路径（特别是 `sample_by=user`）
   - 便于调参（`alpha/beta/weight/tau/topk/layers`）

---

## 3. 采样与分组：先解决“按 item 训练”信息丢失问题

原先 `sample_by=item` 时，会把所有 sampled users 直接拼成一个 `x`，导致 loss 看不到“每个 user 属于哪个 item”。

为此实现了两种采样函数：

- `_sample_users_from_items(...)`
- `_sample_users_from_items_with_dedup(...)`（默认推荐）

它们都会返回：

1. `x`：用于模型前向的 user embedding 张量
2. `item_groups`：每个 item 在 `x` 中对应的行索引集合

### 3.1 去重逻辑（重点）

`_sample_users_from_items_with_dedup(...)` 做了“跨 item 去重”：

- 若某个 user 同时出现在多个 sampled item 中，只在 `x` 里保留一份 embedding
- 但在 `item_groups` 里，这个 user 仍可出现在多个 item 的组中（通过同一个行索引复用）

这恰好满足你的需求：

- 输入特征不重复编码（效率更高，符合“重复用户剔除”）
- 条件熵统计仍保留 item 归属关系（可计算 $H(C|I)$）

---

## 4. 正则具体实现：`_compute_item_mi_regularizer(...)`

位置：`train_user.py` 中新增函数 `_compute_item_mi_regularizer(...)`。

输入核心是：

- `residuals`：来自 `model.get_semantic_ids(...)` 的每层残差表示
- `item_groups`：每个 item 的 user 行索引集合
- `mi_alpha, mi_beta, mi_tau, mi_topk, mi_reg_layers`

### 4.1 每层软分配 $\pi_{u,j}$

对第 $k$ 层：

1. 取该层残差 `residual_k`
2. 取该层码本 `codebook = layer.out_proj(layer.embedding.weight)`
3. 计算 L2 距离矩阵 `dist[u, j]`
4. 用 softmax 得到软分配概率：
   - 若 `mi_topk` 生效：仅对 top-k 最近码字做 softmax（其余为 0）
   - 否则：对全部 code 做 softmax

温度由 `mi_tau` 控制（越小越尖锐，越接近 hard assignment）。

### 4.2 全局熵 $H(C)$

先对 batch 所有 user 平均得到全局分布：

$$
\hat p(j)=\frac{1}{B}\sum_u \pi_{u,j}
$$

再计算：

$$
H(C)=-\sum_j \hat p(j)\log(\hat p(j)+\epsilon)
$$

### 4.3 条件熵 $H(C|I)$

对每个 item 组 $U_i$：

$$
\hat p_i(j)=\frac{1}{|U_i|}\sum_{u\in U_i}\pi_{u,j}
$$
$$
H(C|I=i)=-\sum_j\hat p_i(j)\log(\hat p_i(j)+\epsilon)
$$

然后按组大小加权：

$$
H(C|I)=\sum_i \frac{|U_i|}{B_{\text{eff}}}H(C|I=i)
$$

其中 $B_{\text{eff}}=\sum_i |U_i|$（组成员总数；在有重叠 membership 时是“membership 数”而非 unique user 数）。

### 4.4 每层正则与多层求和

每层：

$$
\mathcal{L}^{(k)}_{reg}=\alpha H(C^{(k)}|I)-\beta H(C^{(k)})
$$

跨层累加得到 `reg_total`，同时返回平均 `mi_h_global` / `mi_h_cond` 作为日志指标。

---

## 5. 训练循环如何接入

在 `_train_rqvae(...)` 中，训练逻辑变成：

1. 采样得到 `x` 与 `item_groups`（item 模式）
2. 调 `model.get_semantic_ids(x, gumbel_t=0.2)` 得到量化输出
3. 用 decode + reconstruction 计算 `reconstruction_loss`
4. 取 `quantized.quantize_loss` 作为 VQ loss
5. `base_loss = mean(reconstruction + quantize)`
6. 若启用 item MI 正则：
   - `mi_reg, mi_metrics = _compute_item_mi_regularizer(...)`
7. `total_loss = base_loss + mi_weight * mi_reg`
8. `total_loss.backward(); optimizer.step()`

也就是说，正则是“旁路加法”，不会破坏原始 RQ-VAE 基础损失。

---

## 6. 新增参数与默认值

`train_user.py` 新增参数：

- `--enable-item-mi-loss`：开启该正则（默认关闭）
- `--mi-alpha`：条件熵权重，默认 `1.0`
- `--mi-beta`：全局熵权重，默认 `1.0`
- `--mi-weight`：总正则缩放系数，默认 `1.0`
- `--mi-tau`：软分配温度，默认 `0.2`
- `--mi-topk`：top-k 近邻码字，默认 `32`（<=0 或 >= codebook_size 时等价全量）
- `--mi-reg-layers`：参与正则的层数，默认 `3`
- `--no-dedup-users-in-item-batch`：关闭 item batch 去重（默认是去重开启）

---

## 7. 行为预期与调参建议

### 7.1 预期趋势

开启后通常希望看到：

- `mi_h_cond` 逐步下降（item 内更集中）
- `mi_h_global` 不塌缩到很低（整体仍有分散性）
- `mi_reg` 与 `base_loss` 同数量级或略小（避免压过重建目标）

### 7.2 建议起点

可先从这组参数试：

- `--enable-item-mi-loss`
- `--sample-by item`
- `--users-per-item 2~4`
- `--mi-alpha 1.0`
- `--mi-beta 0.5~1.0`
- `--mi-weight 0.1~0.5`（建议先小）
- `--mi-tau 0.2~0.5`
- `--mi-topk 16~64`

### 7.3 常见现象

1. `mi_h_global` 下降过快（码本趋同）  
   - 增大 `mi_beta`
   - 减小 `mi_weight`
   - 增大 `mi_topk` 或 `mi_tau`

2. `mi_h_cond` 降不下来（item 内不够集中）  
   - 增大 `mi_alpha`
   - 适当减小 `mi_tau`
   - 增加 `users_per_item` 让每个 item 条件分布估计更稳定

3. 训练不稳定  
   - 减小 `mi_weight`
   - 先固定 `mi_topk` 在中等值（如 32）
   - 观察 base loss 是否被正则压制

---

## 8. 与你的需求一一对应关系

你的需求 | 实现方式
---|---
同 item 用户更集中 | 最小化 $H(C|I)$
不同 item 相对分散 | 最大化 $H(C)$（通过 `-beta * H(C)`）
同 semantic id 可对应多个 user | soft assignment + 分布熵约束本身允许 many-to-one 聚类
item 采样并去重重复 user | `_sample_users_from_items_with_dedup(...)`
保留全局熵与局部熵统计 | `item_groups` + `mi_h_global/mi_h_cond` 日志

---

## 9. 一句话总结

当前实现是在 `train_user.py` 中通过“**去重但保留 item 分组** + **分层软分配熵正则**”实现 item-conditioned user clustering，不改 `QuantizeLoss` 本体，而是在总损失上做可控叠加。
