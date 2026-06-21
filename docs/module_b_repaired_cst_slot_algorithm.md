# Module B：二进制侧 CST/Slot 结构解析器

本节给出修订后的 Module B 设计。原 Module B 以二进制-源码跨图对齐为主，但推理阶段不存在源码侧 AST/CFG，因此无法依赖对齐矩阵 `M` 构建骨架。修订后的 Module B 将骨架恢复建模为二进制侧结构预测任务：模型仅使用二进制 CFG、Module A 的 BB 表示与模式概率，直接预测 canonical CST、slot 结构与 BB-to-slot grounding。

```text
旧设计：binary + source -> BB-source alignment -> rule-based skeleton
新设计：binary only -> canonical CST + slots + BB-to-slot grounding
```

---

## 6.1 模块定义

Module B 是 StrucLift 的结构解析层。其推理输入为：

```text
H_b      Module A 输出的 BB 表示
P_pat    Module A 输出的 soft pattern distribution
E_cfg    二进制 CFG 边
F_cfg    BB 级 CFG 结构特征
R_cfg    CFG pairwise 关系：edge、dominator、postdominator、backedge、distance、reachability
```

训练期额外使用 `source AST` 与 `M_gt` 构造 teacher，但它们不作为推理输入。

Module B 输出：

```text
1. CST outline
   控制结构数量、类型序列与嵌套深度。

2. Control region
   每个控制 anchor 覆盖的 BB 集合。

3. Parent / role
   控制结构父子关系，以及 cond / then / else / loop_body / stmt 等角色。

4. Slot grounding
   FILL_COND、FILL_BLOCK、FILL_BRANCH 及其对应 BB。

5. Function-internal slotted skeleton
   函数体内部控制结构与 FILL 槽位。
```

函数声明不属于 Module B 的预测目标。训练与审计文本可从源码复制并规范化函数声明，但 Module B 仅预测函数体内部 skeleton。

---

## 6.2 Teacher 构造

训练期利用源码构造监督信号，但推理期不依赖源码。teacher 构造遵循：

```text
source AST -> canonical CST structure
M_gt       -> CST / slot to BB grounding
```

源码 AST 用于定义控制结构形状，`M_gt` 用于确定每个控制节点或 slot 对应的 BB。

Canonical CST 保留以下控制节点：

```text
FUNC
IF
LOOP     统一表示 for / while / do-while
SWITCH
SLOT
```

slot 类型包括：

```text
FILL_BLOCK
FILL_COND
FILL_BRANCH
slot_000  函数体内部 preamble/context 保留槽
```

对 BB `b_i` 与 slot `s_k`，定义 grounding 质量：

```text
q_slot[i, k] =
    sum M_gt[i, j]
    for every AST node j belonging to slot s_k
```

BB 的主 slot 标签为：

```text
y_slot[i] = argmax_k q_slot[i, k]
```

对控制节点 `c_k`，region 标签为：

```text
y_region[i, k] = 1
    if sum M_gt[i, j] over AST nodes of control c_k > tau
    else 0
```

最终导出的训练标签包括：

```text
y_ctrl       BB 是否为 IF / LOOP / SWITCH / NONE anchor
y_region     控制 anchor 覆盖的 BB 集合
y_parent     CST 父节点
y_role       BB 在父结构下的角色
y_slot       BB 的 slot 类型
y_same_slot  BB pair 是否属于同一 slot
y_slot_ot    BB-to-slot 软分配监督
y_outline    count / kind / depth 控制结构主干
```

---

## 6.3 模型结构

### 6.3.1 CFG Structural Transformer

Module B 首先使用结构感知 Graph Transformer 编码二进制 CFG。每个 BB 的初始表示为：

```text
x_i = concat(
    h_b[i],
    W_p * P_pat[i],
    f_cfg[i]
)
```

结构偏置注意力定义为：

```text
score_ij^h =
      dot(W_Q^h x_i, W_K^h x_j) / sqrt(d_h)
    + b_edge^h(i, j)
    + b_dom^h(i, j)
    + b_postdom^h(i, j)
    + b_back^h(i, j)
    + b_reach^h(i, j)
    + b_dist^h(i, j)

alpha_ij^h = softmax_j(score_ij^h)

z_i = FFN(
    x_i + concat_over_heads(
        sum_j alpha_ij^h * W_V^h x_j
    )
)
```

该编码器使 BB 表示显式感知支配关系、回边、可达性与拓扑距离。

### 6.3.2 CST Outline Decoder

Outline decoder 预测控制结构主干：

```text
N_ctrl   控制节点数量
t_k      第 k 个控制节点类型：IF / LOOP / SWITCH / NONE
d_k      第 k 个控制节点嵌套深度
```

设最多预测 `K` 个控制节点。对每个位置 `k`，使用可学习 query 从 BB 表示中读取上下文：

```text
u_k = AttnQuery(p_k, {z_i})
u_1:K = GRU(u_1:K)

p(N_ctrl | G_b) = softmax(W_N * pooled_z)
p(t_k | G_b)    = softmax(W_t * u_k)
p(d_k | G_b)    = softmax(W_d * u_k)
```

### 6.3.3 Region Proposal

对每个候选控制 anchor `i`，预测控制类型：

```text
p_ctrl[i] = softmax(W_ctrl * z_i)
```

并预测 anchor `i` 是否覆盖 BB `j`：

```text
p_region[i, j] = sigmoid(
    MLP_region(concat(z_i, z_j, f_cfg_pair[i, j]))
)
```

Region proposal 用于确定控制结构覆盖范围，并为后续 role 与 slot grounding 提供边界。

### 6.3.4 Parent / Role Parser

CST 嵌套关系建模为有根依存树。父节点分数为：

```text
s_parent[i, j] =
      z_i^T U z_j
    + MLP_parent(concat(z_i, z_j, f_cfg_pair[i, j]))

s_root[i] = W_r * z_i
```

Role parser 预测 BB 或节点在父结构下的角色：

```text
cond / then / else / loop_body / switch_body / stmt / body / none
```

节点级 role：

```text
p_role[i] = softmax(W_role * z_i)
```

parent-child pair role：

```text
p_role_pair[i, j] = softmax(
    MLP_role(concat(z_i, z_j, f_cfg_pair[i, j]))
)
```

### 6.3.5 Slot Grounding 与 OT

每个 BB 的 slot 类型分布为：

```text
p_slot[i] = softmax(W_slot * z_i)
```

同时预测 BB pair 是否属于同一 slot：

```text
p_same_slot[i, j] = sigmoid(
    MLP_same(concat(z_i, z_j, f_cfg_pair[i, j]))
)
```

为获得稳定的 BB-to-slot 软分配，本文将 OT 用于 BB-to-slot grounding。设 slot anchor 为 `k`，匹配分数为：

```text
s_slot[i, k] = MLP_anchor(
    concat(z_i, z_k, f_cfg_pair[i, k])
)

Q = Sinkhorn(S_slot, mu, nu)
```

其中 `mu` 为 BB 侧边际分布，`nu` 由 teacher slot size 估计。与旧的 BB-to-source OT 不同，该分支在推理期仍然可用，因为 slot 是 Module B 自身预测对象。

---

## 6.4 Skeleton 序列化

Module B 输出函数体内部 slotted skeleton，例如：

```c
<FILL_BLOCK:slot_000>;
while (<FILL_COND:slot_001>) {
    if (<FILL_COND:slot_002>) {
        <FILL_BLOCK:slot_003>;
    }
}
```

训练与审计时可以由数据构建阶段复制源码函数声明进行包装：

```c
int func(int argc, char **argv) {
    <FILL_BLOCK:slot_000>;
    while (<FILL_COND:slot_001>) {
        <FILL_BLOCK:slot_002>;
    }
}
```

Module B 给 Module C 的实际输入应是结构化对象：

```json
{
  "decl": "int func(int argc, char **argv)",
  "skeleton_body": ["<FILL_BLOCK:slot_000>;", "..."],
  "slots": [
    {
      "slot_id": "slot_000",
      "slot_type": "FILL_BLOCK",
      "role": "preamble",
      "bbs": [0, 1],
      "bin_ctx": ["..."]
    }
  ]
}
```

Module C 只填充 slot 内容，不重新生成控制结构主干。

---

## 6.5 训练目标

算法层面将 Module B 的监督归纳为五组结构目标：

```text
L_B =
      lambda_cst    * L_cst
    + lambda_region * L_region
    + lambda_slot   * L_slot
    + lambda_ot     * L_ot
    + lambda_skel   * L_skel
```

CST structure loss：

```text
L_cst =
      L_count
    + L_kind
    + L_depth
    + L_parent
    + L_tree
```

Region grounding loss：

```text
L_region =
      L_ctrl
    + L_region_pair
```

Slot structure loss：

```text
L_slot =
      L_role
    + L_slot_type
    + L_same_slot
    + L_role_pair
```

OT grounding loss：

```text
L_ot = KL(Q_gt || Q)
```

Skeleton reconstruction proxy：

```text
L_skel =
      eta_cst  * L_cst
    + eta_slot * L_slot
    + eta_ot   * L_ot
```

---

## 6.6 训练与推理流程

训练流程：

```text
1. 使用 source AST + M_gt 导出 B_skel teacher。
2. 加载 Module A checkpoint，得到 H_b 与 P_pat。
3. 训练 BinaryCSTParser。
4. 使用 grouped structural loss 优化 CST、region、slot 与 BB-to-slot grounding。
5. 保存 best_bskel.pt。
```

推理流程：

```text
1. binary function -> CFG / BB instruction sequence
2. Module A -> H_b, P_pat
3. Module B -> CST outline + region + parent/role + slots + BB-to-slot
4. serialize -> function-internal slotted skeleton
5. Module C -> fill slots
6. optional Module D -> structure reward refinement
```

推理阶段不使用 source AST、source CFG 或 `M_gt`。
