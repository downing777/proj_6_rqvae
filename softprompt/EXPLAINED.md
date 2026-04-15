# SID 个性化标题训练详解

这份文档详细解释 `softprompt` 目录里的训练逻辑、关键公式、为什么这样设计，以及每一步在代码中的对应位置。

---

## 1. 任务定义

你已经有了用户 SID（例如三元组 `(1, 2, 3)`），目标是：

- 对同一个商品，针对不同 SID 生成不同标题；
- 同一 SID 对应相近偏好人群，标题风格应稳定；
- 在线上优先输出单条“最优标题”（而不是多样化采样）。

所以我们建模为条件生成：

$$
\text{title} \sim p_\theta(y \mid x, s)
$$

- $x$: 商品上下文（品牌、卖点、属性等）
- $s$: SID（三层离散 ID）
- $y$: 目标标题

---

## 2. 为什么不用单一 special token

你最初的想法是把 SID 映射成一个 token 再喂给模型。这个思路可行，但存在信息瓶颈：

- 单一 token 的表达能力有限，难承载细粒度偏好；
- 对复杂语义偏好（价格敏感、颜值导向、功能导向）容易欠拟合。

因此改为 **多虚拟 token 前缀**（prefix soft prompt）：

$$
P_s \in \mathbb{R}^{m \times d}
$$

- $m$: 虚拟 token 数（如 8~32）
- $d$: LLM hidden size

模型看到的是一段“SID 条件前缀 + 商品文本”，表达能力更强且参数高效。

---

## 3. SID -> 前缀向量的参数化（你想法的升级版）

对应代码：`softprompt/models/sid_prefix.py`

### 3.1 SID 嵌入

SID 三元组 $s=(s_1,s_2,s_3)$，每层各自 embedding：

$$
e_i = \mathrm{Emb}_i(s_i), \quad i\in\{1,2,3\}
$$

拼接后得到：

$$
h_s = \mathrm{MLP}([e_1;e_2;e_3])
$$

### 3.2 Basis mixing（保留你的 $P \cdot T$ 思路）

我们定义：

- 可学习 basis 矩阵 $T \in \mathbb{R}^{K \times d}$
- 由 SID 生成的权重 $W_s \in \mathbb{R}^{m \times K}$

其中

$$
W_s = \mathrm{reshape}(\mathrm{Linear}(h_s), m, K), \quad
\alpha_s = \mathrm{softmax}(W_s, \text{dim}=K)
$$

最终前缀：

$$
P_s = \alpha_s T
$$

这就是你原先 `SID = P*T` 的多 token 版本（不是单 token），和计划一致。

---

## 4. 条件注入到语言模型

对应代码：`softprompt/models/wrapper.py`

给定文本 token embedding $E_x \in \mathbb{R}^{n\times d}$，SID 前缀 $P_s\in\mathbb{R}^{m\times d}$，拼接：

$$
E' = [P_s; E_x]
$$

attention mask 同步补齐前缀位置为 1。

训练时 labels 的前缀段设为 `-100`，不参与 CE 损失，只学习正文标题 token。

---

## 5. 两阶段训练：SFT -> DPO

---

### 5.1 阶段一：SFT（监督微调）

对应代码：

- `softprompt/train/common.py`
- `softprompt/train/train_sft.py`

数据格式（每行 JSON）：

- `item_id`
- `sid`: `[rqid_0, rqid_1, rqid_2]`
- `context`
- `target_title`

目标是最小化负对数似然（交叉熵）：

$$
\mathcal{L}_{\text{SFT}}
=
\mathbb{E}_{(x,s,y)}
\left[
-\sum_{t} \log p_\theta(y_t \mid y_{<t}, x, s)
\right]
$$

作用：

- 先让模型“会写”且与商品强相关；
- 给 DPO 一个稳定初始化，避免直接偏好优化导致语义漂移。

---

### 5.2 阶段二：DPO（偏好对齐）

对应代码：`softprompt/train/train_dpo.py`

数据格式（每行 JSON）：

- `item_id`
- `sid`
- `context`
- `title_chosen`
- `title_rejected`
- `negative_type`（`hard_cross_sid` 或 `easy_random`）

对每个样本，定义序列 log-prob：

$$
\log \pi_\theta(y\mid x,s)=\sum_t \log p_\theta(y_t\mid y_{<t},x,s)
$$

策略模型（policy）与参考模型（reference）分别记为 $\pi_\theta, \pi_{\text{ref}}$。

DPO 损失：

$$
\mathcal{L}_{\text{DPO}}
=
-\log \sigma\left(
\beta\left[
(\log \pi_\theta(y^+\!\mid x,s)-\log \pi_\theta(y^-\!\mid x,s))
-
(\log \pi_{\text{ref}}(y^+\!\mid x,s)-\log \pi_{\text{ref}}(y^-\!\mid x,s))
\right]
\right)
$$

- $y^+$: chosen 标题
- $y^-$: rejected 标题
- $\beta$: 偏好区分强度

直觉：让 policy 相比 reference 更偏向 chosen。

---

## 6. 可选 KL 正则（防止偏离商品事实）

在 `train_dpo.py` 里实现了可选 `--kl-to-nosid-weight`。

思想是控制“带 SID”输出分布不要过度偏离“无 SID/弱条件”输出，缓解：

- 过度追求点击性导致事实偏移；
- 标题风格过分激进。

形式上可写作：

$$
\mathcal{L}
=
\mathcal{L}_{\text{DPO}}
+
\lambda_{\text{KL}}\cdot
\mathrm{KL}\left(
p_\theta(\cdot\mid x,s)
\parallel
p_\theta(\cdot\mid x,\varnothing)
\right)
$$

---

## 7. 负样本设计（格式层面）

对应脚本：`softprompt/data/build_dpo_pairs.py`

文档里定义了两类负样本字段：

- `hard_cross_sid`: 同商品下来自其他 SID 的高分标题（更难）
- `easy_random`: 随机低质量标题（更易）

推荐训练混比（计划中的默认）：

$$
\text{hard}:\text{easy}=7:3
$$

理由：hard negative 能学到 SID 区分边界，easy negative 保证基础鲁棒性。

---

## 8. 推理与线上单条最优标题

对应代码：`softprompt/infer/generate_title.py`

输入 `(item_id, sid, context)`，流程：

1. 由 SID 生成前缀 $P_s$
2. 与 prompt embedding 拼接
3. `generate` 输出标题

为了“单条最优”，建议线上：

- `num_beams > 1`（如 4）用于稳定；
- 或 `temperature` 低值（如 0.2~0.7）减少随机波动。

---

## 9. 离线评估与回退

对应代码：`softprompt/eval/offline_eval.py`

已实现指标：

- `pair_win_rate`: chosen 对 rejected 胜率
- `keyword_recall`: 标题对上下文关键词覆盖
- `distinct_1/2`: 去重与多样性
- `fallback_ratio`: 回退比例

### 回退门控

定义 margin：

$$
\Delta = \log \pi_\theta(y^+\mid x,s)-\log \pi_\theta(y^-\mid x,s)
$$

若 $\Delta < \tau$（`--min-margin`），说明模型置信度不足，回退到安全标题（例如 chosen 模板或默认标题）。

---

## 10. 参数建议（起步）

- `num_virtual_tokens`: 16（表达力和稳定性平衡）
- `num_basis_tokens`: 64
- `sid_embed_dim`: 64
- `beta` (DPO): 0.05 ~ 0.2
- `kl_to_nosid_weight`: 0 ~ 0.1（先从 0 开始）
- 训练策略：先 `--freeze-backbone`，稳定后再尝试 LoRA 放开少量层

---

## 11. 代码映射总览

- SID 前缀编码：`softprompt/models/sid_prefix.py`
- LLM 注入包装：`softprompt/models/wrapper.py`
- 数据格式定义：`softprompt/data/build_dpo_pairs.py`
- SFT 训练：`softprompt/train/train_sft.py`
- DPO 训练：`softprompt/train/train_dpo.py`
- 推理生成：`softprompt/infer/generate_title.py`
- 评估与回退：`softprompt/eval/offline_eval.py`

---

## 12. 一句话总结

这套训练本质是：用 SID 生成一段可学习“人群偏好前缀”，先用 SFT 学会“写对标题”，再用 DPO 学会“更偏向该 SID 喜欢的标题”，最后用 margin 回退保证线上稳定性和安全性。
