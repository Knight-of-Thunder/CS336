# Resource Computing

## 一、计算 FLOPs 数量（单个 Batch）

### 1. 模型组成与计算项分析

- **Embedding 层：** 查表操作（可忽略 FLOPs）
- **Transformer Block：**
  - **Multi-Head Self-Attention (MHSA)：**
    - 三次线性投影（Q, K, V）： `6 · S · D²`
    - 每个头的注意力得分计算 (QₕKₕᵀ)： `2 · S² · D`
    - 注意力结果与 Vₕ 的乘法： `2 · S² · D`
  - **Feed-Forward Network (FFN, 使用 GLU)：**
    - 三个线性层： `6 · S · D · F`
- **LM Head（反向 Embedding 操作）：** `2 · S · D · V`

### 2. 总 FLOPs 公式

$$
\mathbf{F} = \mathbf{L} \cdot 
\left(
\underbrace{8 S D^2 + 4 S^2 D}_{\text{Self-Attention}} + 
\underbrace{6 S D F}_{\text{FFN (GLU)}}
\right)
+ 
\underbrace{2 S D V}_{\text{LM Head}}
$$

---

## 二、计算参数内存占用量

- **Embedding 与 LM Head 权重共享：** `D · V`
- **每个 Block：**
  - Norm1： `D`
  - MHSA： `4D²`
  - Norm2： `D`
  - FFN： `3D · F`

**总参数量：**

$$
\text{Params} = L(2D + 4D^2 + 3D F) + D + D V
$$

---

## 三、数值计算示例：GPT-2 XL

**配置参数：**

| 参数 | 值 |
| ---- | ---- |
| vocab_size (V) | 50,257 |
| context_length (S) | 1,024 |
| num_layers (L) | 48 |
| hidden_dim (D) | 1,600 |
| num_heads | 25 |
| ff_dim (F) | 6,400 |

**参数内存计算：**

$$
48 \times (2 \times 1600 + 4 \times 1600^2 + 3 \times 1600 \times 6400) + 1600 + 1600 \times 50257 = 2.28788 \times 10^9\ \text{Bytes}
$$

换算为 GB：

$$
2.28788 \times 10^9 / 1024^3 \approx 2.13\ \text{GB}
$$

---

## 四、问题回答

### (b) GPT-2 XL 的矩阵乘法与 FLOPs

- 包含所有 Q、K、V 投影、Attention 矩阵乘法、FFN 三层线性计算及输出头线性层。
- 使用上式计算可得总 FLOPs 数量。

---

### (c) 哪些部分消耗最多 FLOPs？

从公式可知：

- Attention 模块： `O(L(8S D² + 4S² D))`
- FFN 模块： `O(L · 6S D F)`  
  若代入 `F = 4D`，则 FFN ≈ `O(24S D²)`，其常数项远大于 Attention，因此 **FFN 是主要的 FLOPs 消耗来源**。

---

### (d) 各 GPT-2 变体 FLOPs 对比分析

| 模型 | 总 FLOPs | Self-Attn 占比 | FFN 占比 | LM Head 占比 |
|:--|:--|:--|:--|:--|
| GPT-2 Small (L=12, D=768) | 349.5 GFLOPs (0.35 TFLOPs) | 96.6 G (27.6%) | 173.9 G (49.8%) | 79.0 G (22.6%) |
| GPT-2 Medium (L=24, D=1024) | 1033.1 GFLOPs (1.03 TFLOPs) | 309.2 G (29.9%) | 618.5 G (59.9%) | 105.4 G (10.2%) |
| GPT-2 Large (L=36, D=1280) | 2257.6 GFLOPs (2.26 TFLOPs) | 676.4 G (30.0%) | 1449.5 G (64.2%) | 131.7 G (5.8%) |
| GPT-2 XL (L=48, D=1600) | 4513.3 GFLOPs (4.51 TFLOPs) | 1328.7 G (29.4%) | 3019.9 G (66.9%) | 164.7 G (3.7%) |

> 注：GPT-2 XL 的计算已基于 GLU 版本 FFN ($6SDF$) 修正。

---

### 结论：模型规模对 FLOPs 占比的影响

随着模型规模（L 与 D）增大：

- **FFN 部分的 FLOPs 占比持续上升。**  
  - **原因：** FFN 的复杂度随 $O(L D^2)$ 增长，且常数项（24）显著大于 Attention 部分（8），因此其在总计算量中的主导地位更明显。
- **LM Head 占比逐渐下降。**  
  - 随着 $D$ 增大，LM Head ($2SDV$) 的增长速度相对较慢。
- **Self-Attention 占比保持相对稳定。**

---
