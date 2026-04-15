统一用 code 索引 $j$，item 索引 $i$，用户索引 $u$；第 $k$ 级 codebook 大小记为 $M_k$

---

## 可导 MI-style 正则（不改 RQ-VAE 硬结构）

### 随机变量与目标
- $I$：item 随机变量，取值为具体 item $i$
- $C^{(k)}$：第 $k$ 级 RQ 的离散 code 随机变量，取值为 code 索引 $j\in\{1,\dots,M_k\}$

希望同一 item 的用户在 code 上更集中、同时全局用码不塌缩。用互信息思想形式化：
$$
I(C^{(k)};I)=H(C^{(k)})-H(C^{(k)}\mid I)
$$
最大化互信息等价于增大 $H(C^{(k)})$ 且减小 $H(C^{(k)}\mid I)$。

训练时用两个权重写成正则（最小化）：
$$
\mathcal{L}^{(k)}_{reg}= \alpha\,H(C^{(k)}\mid I)\;-\;\beta\,H(C^{(k)})
$$
总正则对前 $L$ 级求和：$\mathcal{L}_{reg}=\sum_{k=1}^{L}\mathcal{L}^{(k)}_{reg}$。

---

### 为什么 $H(C^{(k)}\mid I)$ 要按 item 加权
条件熵定义为对 $I$ 的期望：
$$
H(C^{(k)}\mid I)=\sum_i p(i)\,H(C^{(k)}\mid I=i)
$$
一个 batch 有 $B$ 条交互样本 $(u,i)$。令 $U_i$ 表示 batch 中 item $i$ 对应的用户集合，则经验频率估计：
$$
\hat p(i)=\frac{|U_i|}{B}
$$
因此 batch 内条件熵估计为：
$$
\widehat{H}(C^{(k)}\mid I)=\sum_i \frac{|U_i|}{B}\;\widehat{H}(C^{(k)}\mid I=i)
$$
这就是权重取 $|U_i|/B$ 的原因（它是在近似 $p(i)$）。

---

## 关键实现：hard RQ 前向不变，旁路 soft assignment 让正则可导
RQ-VAE 的离散取码（argmin 最近邻）、STE/EMA、以及输出 meta-user id 全部保持不变。

为了让熵可导，仅对正则项并行计算一个 soft assignment。对每个用户 $u$ 和每个级别 $k\le L$：

1) 取该级残差向量 $r^{(k)}_u$（按你们现有 RQ 实现）。  
2) 计算到码字 $e^{(k)}_j$ 的距离：
$$
d^{(k)}_{u,j}=\|r^{(k)}_u-e^{(k)}_j\|^2
$$
3) 取 top‑t 最近码字集合 $N_t(u,k)\subset\{1,\dots,M_k\}$（t=16/32/64）。  
4) 在候选集合上 softmax 得到概率（其余 $j$ 概率为 0）：
$$
\pi^{(k)}_{u,j}=\mathrm{softmax}_{j\in N_t(u,k)}\left(-d^{(k)}_{u,j}/\tau\right)
$$
$\tau$ 固定即可（如 0.1~1）。  
注意：hard code id 仍由 argmin 得到；$\pi$ 仅用于计算熵并反传。

---

## 用 soft assignment 估计分布与熵（按级别计算）
对每个级别 $k\le L$：

### 1) 全局分布与全局熵
$$
\hat p^{(k)}(j)=\frac{1}{B}\sum_{u\in \text{batch}}\pi^{(k)}_{u,j}
$$
$$
\widehat{H}(C^{(k)})= -\sum_{j\in S^{(k)}} \hat p^{(k)}(j)\log(\hat p^{(k)}(j)+\epsilon)
$$
其中 $S^{(k)}=\{j:\hat p^{(k)}(j)>0\}$（batch 激活到的稀疏支撑集），$\epsilon=10^{-9}$。

（可选：对 $\hat p^{(k)}$ 做 EMA 平滑后再算 $\widehat{H}(C^{(k)})$，降低 batch 抖动。）

### 2) item 条件分布与条件熵
对 batch 中每个 item $i$，其用户集合 $U_i$：
$$
\hat p^{(k)}_i(j)=\frac{1}{|U_i|}\sum_{u\in U_i}\pi^{(k)}_{u,j}
$$
$$
\widehat{H}(C^{(k)}\mid I=i)= -\sum_{j\in S^{(k)}_i}\hat p^{(k)}_i(j)\log(\hat p^{(k)}_i(j)+\epsilon)
$$
其中 $S^{(k)}_i=\{j:\hat p^{(k)}_i(j)>0\}$。

按条件熵定义做加权：
$$
\widehat{H}(C^{(k)}\mid I)=\sum_i \frac{|U_i|}{B}\;\widehat{H}(C^{(k)}\mid I=i)
$$

### 3) 该级正则与总正则
$$
\mathcal{L}^{(k)}_{reg}= \alpha\,\widehat{H}(C^{(k)}\mid I)\;-\;\beta\,\widehat{H}(C^{(k)})
$$
$$
\mathcal{L}_{reg}=\sum_{k=1}^{L}\mathcal{L}^{(k)}_{reg}
$$

---

## 总损失与推理
$$
\mathcal{L}= \mathcal{L}_{base(RQ\text{-}VAE)} + \mathcal{L}_{reg}
$$
推理时只输出 hard argmin 得到的前 $L$ 级 code 索引作为 meta-user id；soft assignment 不参与推理。

---