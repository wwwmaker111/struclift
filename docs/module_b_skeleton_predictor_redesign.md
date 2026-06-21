# Module B 修复方案：从跨图对齐到二进制侧骨架预测

> 目的：修复当前 Module B “训练期依赖源码对齐、推理期却没有源码”的逻辑断点，让骨架恢复变成一个可以训练、可以评估、可以部署的二进制侧结构预测任务。

本文档是对原方案第六章 Module B、6.5 骨架生成、6.6 损失函数、9.3 Stage 2、9.4 Stage 3 的修订建议。它不推翻原来的 Module A/C/D，而是把 Module B 从“只做跨图对齐”改成“以骨架预测为主、跨图对齐为辅助”。

---

# 1. 当前问题

## 1.1 原 Module B 的设计断点

原文中 Module B 的输入包含：

```text
二进制侧：H_b, h_b, P_pat, E_b
源码侧：  H_s, h_s, AST 类型 T_s, E_s
输出：    对齐矩阵 M、增强表示 H_hat、结构骨架 Skeleton
```

这个设计在训练期可以跑，因为训练数据有源码；但是推理期没有源码侧图，所以不能真正输出源码侧对齐矩阵 `M`。如果骨架生成依赖 `M`，那推理阶段就缺少关键输入。

更具体地说，当前逻辑是：

```text
训练期：binary + source -> Module B -> M -> skeleton
推理期：binary only     -> 没有 source -> 没有 M -> 只能退化成 P_pat + CFG rule
```

所以实际部署时，Module B 训练出来的“跨图对齐能力”并没有稳定转化成“无源码骨架恢复能力”。

## 1.2 当前规则式骨架 builder 的不足

当前骨架主要依赖：

```text
P_pat argmax + binary CFG + dominator/postdominator rule -> skeleton
```

这类规则能恢复一部分明显结构，例如简单 loop、简单 if，但在下面场景会不稳定：

| 问题 | 根因 |
|---|---|
| false if header | Module A 的 if_header 概率高，但该 BB 可能只是短路条件、跳转整理或编译器生成的比较块 |
| 平级结构误嵌套 | CFG 中两个分支的 postdominator / join 关系不总能唯一决定源码写法 |
| if / else 形态差异 | `if-return; stmt;` 与 `if-return else stmt` 在二进制 CFG 上可能语义等价，但源码 AST 形式不同 |
| switch 恢复不足 | switch_case_body 标签稀疏，jump table / if-chain lowering 会让源码 switch 与二进制形态差距很大 |
| body slot 不稳定 | 一个源码语句可对应多个 BB，一个 BB 也可能覆盖多条源码语句；一 BB 一 slot 不合理 |

因此，继续把核心希望放在纯规则 builder 上，上限会比较低。正确方向应该是：规则只做约束和解码，主能力由监督学习得到。

---

# 2. 修复后的总体思路

## 2.1 一句话版本

把 Module B 拆成两个分支：

```text
B_skel：主分支。只输入二进制侧 H_b + P_pat + CFG，直接预测 CST / slot / BB-to-slot。
B_align：辅助分支。训练期可输入源码侧图学习 BB <-> AST/语句对齐，但推理期不依赖它。
```

修复后的主链路是：

```text
训练期：
source AST + M_gt -> canonical CST teacher / slot teacher
binary CFG + Module A outputs -> B_skel -> predicted CST / slot
loss(predicted CST, teacher CST)

推理期：
binary CFG + Module A outputs -> B_skel -> skeleton + slots + slot_to_BB
Module C 按 slot 填内容
```

也就是说，源码不再是 Module B 推理时的输入，而是训练期生成监督标签的 teacher。

## 2.2 与原四模块的关系

| 模块 | 原作用 | 修复后作用 |
|---|---|---|
| Module A | 学 BB 表示和 pattern label | 保留。输出 `H_b` 和 soft `P_pat`，但不把 argmax 当绝对真理 |
| Module B | 跨图对齐 + rule skeleton | 改为 `B_skel` 直接预测二进制侧 CST；`B_align` 只做可选辅助 |
| Module C | 在骨架里填表达式，输入 Skeleton + H_hat + M | 改为输入 Skeleton + slots + slot_to_BB + per-slot binary context |
| Module D | 用结构一致性 RL 优化 | 改为基于 `B_skel` 的 canonical CST 做结构奖励；默认冻结 B_skel，只更新 C |

---

# 3. Canonical CST 定义

## 3.1 为什么不用 AST exact

源码 AST 和二进制可恢复的控制结构不是天然等价的。例如：

```c
if (x) {
    return;
}
foo();
```

和：

```c
if (x) {
    return;
} else {
    foo();
}
```

这两段语义可能等价，但 AST 不一样。再比如 `for`、`while`、`do while` 在二进制里常常都表现为 loop + condition + backedge。

所以 Module B 不应该追求源码 AST 字面完全一致，而应该预测一个 canonical CST，也就是“规范化控制结构树”。

## 3.2 CST 节点类型

Canonical CST 只保留对后续生成最重要的控制结构：

```text
FUNC       函数根节点
SEQ        顺序块容器
IF         条件分支
LOOP       循环，统一表示 for / while / do-while
SWITCH     多路分支
CASE       switch case
SLOT       待填槽位
```

## 3.3 Slot 类型

槽位不再是一 BB 一句，而是 region-level slot：

```text
FILL_ENTRY   函数开头的普通声明/初始化区域，可为空
FILL_COND    if / loop / switch 的条件表达式
FILL_BLOCK   then / else / loop body / 普通顺序语句块
FILL_CASE    case label 或 case body
FILL_RETURN  return 表达式或 return 语句
```

每个 slot 可以对应多个 BB：

```text
slot_k = {
  slot_type: FILL_BLOCK,
  parent_control: IF_3,
  role: then_body,
  aligned_bbs: [BB_8, BB_9, BB_11],
  binary_context: instructions from BB_8/9/11
}
```

## 3.4 Canonical 序列化规则

Module B 输出的骨架统一序列化为：

```c
<FILL_ENTRY:slot_0>
if (<FILL_COND:slot_1>) {
    <FILL_BLOCK:slot_2>
} else {
    <FILL_BLOCK:slot_3>
}
while (<FILL_COND:slot_4>) {
    <FILL_BLOCK:slot_5>
}
```

关键原则：

1. 条件表达式单独一个槽：`FILL_COND`
2. 控制结构内部内容合成 region slot：`FILL_BLOCK`
3. slot 可以生成多句话
4. `for/while/do-while` 在骨架层统一为 `LOOP`，序列化默认用 `while`
5. 如果训练目标使用 canonical skeleton，则 Module C 的 teacher 也必须按 canonical slot 填充，而不能直接强行用原始完整源码作为逐 token GT

---

# 4. Teacher 构造：训练期如何从源码生成监督

## 4.1 输入

训练期 teacher 构造可以使用源码，因为这是离线标注阶段：

```text
source AST / source CFG
M_gt: BB <-> source AST/statement 的 ground-truth alignment
binary CFG
pattern_labels 或 Module A soft P_pat
```

注意：这里用源码是为了生成训练标签，不是为了推理。

## 4.2 Teacher 构造步骤

### 步骤 1：从源码 AST 得到 canonical CST

将源码 AST 规范化：

```text
ForStmt / WhileStmt / DoStmt -> LOOP
IfStmt                       -> IF
SwitchStmt                   -> SWITCH
CaseStmt                     -> CASE
普通语句序列                 -> FILL_BLOCK
条件表达式                   -> FILL_COND
```

得到：

```text
CST_src = (N_ctrl, N_slot, parent, role, order)
```

其中：

- `N_ctrl` 是控制节点集合，例如 IF、LOOP、SWITCH。
- `N_slot` 是槽位集合，例如 FILL_COND、FILL_BLOCK。
- `parent` 表示树结构。
- `role` 表示节点在父结构下的角色，例如 `cond / then / else / loop_body / case_body / follow`。
- `order` 表示同级顺序。

### 步骤 2：用 M_gt 把 BB 投影到 CST 节点和 slot

对每个 BB_i，根据 `M_gt[i, j]` 找到它主要对应的源码 AST/语句节点 `s_j`，再找到 `s_j` 所属的 canonical CST 节点或 slot：

```math
a(i) = \arg\max_j M^{gt}_{ij}
```

```math
y_i^{slot} = \text{slot\_of}(a(i))
```

如果一个 BB 对齐到多个源码节点，则用质量加权：

```math
\text{score}(i, k) = \sum_{j \in \text{slot}_k} M^{gt}_{ij}
```

```math
y_i^{slot} = \arg\max_k \text{score}(i,k)
```

### 步骤 3：生成 B_skel 的监督标签

对二进制侧每个 BB 或候选控制节点生成：

```text
y_i^ctrl      : BB_i 是否是 IF / LOOP / SWITCH / NONE 的 anchor
y_i^parent    : BB_i 所属父控制节点
y_i^role      : BB_i 在父控制节点下的角色
y_i^boundary  : BB_i 是否是 slot 起点 / 内部 / 外部
y_i^slot      : BB_i 属于哪个 slot
y_i^order     : 同级节点顺序
```

这里的 `M_gt` 只用于 teacher 生成，不作为推理输入。

## 4.3 Teacher 质量过滤

不是所有样本都适合作为强监督。需要给 teacher 打置信度：

| 过滤项 | 处理 |
|---|---|
| source bounds 不完整 | 不进入主训练，只进弱监督或审计 |
| M_gt 行质量过低 | 降低该 BB 的 loss weight |
| AST parse 失败 | 丢弃或只保留 Module A pattern 训练 |
| O2/O3 结构被优化消失 | 标为 unrecoverable，不计入 exact skeleton 主指标 |
| switch lowering 成 if-chain | 保留 semantic label，但 switch exact 指标单独统计 |

每个监督项可有权重：

```math
w_i = w_{\text{align}}(i) \cdot w_{\text{ast}}(i) \cdot w_{\text{opt}}(i)
```

---

# 5. B_skel 模型结构

## 5.1 输入

推理和训练都只给 B_skel 二进制侧信息：

```text
H_b      : Module A 输出的 BB 表示，shape [n_b, d]
P_pat    : Module A 输出的 soft pattern distribution，shape [n_b, C]
E_b      : binary CFG edges
F_cfg    : dominator/postdominator/backedge/outdegree/indegree 等结构特征
```

注意：`P_pat` 是 soft feature，不再只取 argmax。这样可以缓解当前 Module A body 类标签不稳定的问题。

## 5.2 CFG-aware 编码层

先把 BB 表示、pattern 分布和 CFG 特征融合：

```math
x_i = [h_i^b \; || \; W_p P_i^{pat} \; || \; f_i^{cfg}]
```

再用 Graph Transformer / GNN 得到结构上下文化表示：

```math
z_i = \text{CFGEncoder}(x_i, E_b)
```

这里的作用是让模型看到：

- 某个 if_header 的两个 successor 是否真的汇合；
- 某个 loop_header 是否存在 backedge；
- 两个控制节点是平级还是嵌套；
- 某些连续 if_header 是否只是短路条件链。

## 5.3 控制节点预测头

预测每个 BB 是否是控制结构 anchor：

```math
p_i^{ctrl} = \text{softmax}(W_{ctrl} z_i)
```

类别为：

```text
NONE / IF / LOOP / SWITCH
```

损失：

```math
\mathcal{L}_{ctrl}
= - \sum_i w_i^{ctrl} \log p_i^{ctrl}(y_i^{ctrl})
```

为什么需要：Module A 只预测局部 pattern label，B_skel 需要判断这个 header 是否真的应该成为最终 CST 节点。

## 5.4 父节点预测头

对每个候选控制节点或 slot，预测它挂在哪个父控制节点下面：

```math
s_{ij}^{parent}
= \text{MLP}_{parent}([z_i || z_j || f_{ij}^{cfg}])
```

```math
p(parent(i)=j)
= \frac{\exp(s_{ij}^{parent})}{\sum_{k \in \mathcal{C}_i}\exp(s_{ik}^{parent})}
```

其中 `C_i` 是候选父节点集合，包括函数根节点和支配 `i` 的控制节点。

损失：

```math
\mathcal{L}_{parent}
= - \sum_i w_i^{parent} \log p(parent(i)=y_i^{parent})
```

为什么需要：这是修复“平级结构误嵌套”和“嵌套 if 被压扁”的核心。不要只靠手写 postdom rule，而是让模型学习 parent/child 关系。

## 5.5 分支角色预测头

给每个 BB 或 slot 预测它在父控制结构下的角色：

```text
cond / then_body / else_body / loop_body / loop_latch / loop_exit / switch_case / follow / plain
```

```math
p_i^{role}
= \text{softmax}(W_{role}[z_i || z_{parent(i)} || f_{i,parent}^{cfg}])
```

```math
\mathcal{L}_{role}
= - \sum_i w_i^{role} \log p_i^{role}(y_i^{role})
```

为什么需要：同一个 BB 标签为 sequential，并不说明它属于普通顺序块、then 分支、else 分支还是 loop body。必须显式学习 role。

## 5.6 Slot 边界预测头

把连续 BB 合成 region-level slot。使用 BIO 风格标签：

```text
B-COND / I-COND
B-BLOCK / I-BLOCK
B-CASE / I-CASE
O
```

```math
p_i^{bd}
= \text{softmax}(W_{bd}[z_i || P_i^{pat} || f_i^{cfg}])
```

```math
\mathcal{L}_{bd}
= - \sum_i w_i^{bd} \log p_i^{bd}(y_i^{bd})
```

为什么需要：当前“一 BB 一 slot”不合理。slot 边界预测让一个槽位自然覆盖多个 BB。

## 5.7 BB-to-slot 分配头

先为每个预测 slot 得到 slot 向量：

```math
u_k = \text{Pool}(\{z_i : i \in \text{slot candidate } k\})
```

再预测 BB_i 属于 slot_k 的概率：

```math
s_{ik}^{slot}
= \text{MLP}_{slot}([z_i || u_k || f_{ik}^{cfg}])
```

```math
p(slot(i)=k)
= \text{softmax}_k(s_{ik}^{slot})
```

```math
\mathcal{L}_{bb2slot}
= - \sum_i w_i^{slot} \log p(slot(i)=y_i^{slot})
```

为了允许一个 slot 覆盖多个 BB，可以加一个覆盖约束：

```math
\mathcal{L}_{cover}
= \sum_i \left| \sum_k p(slot(i)=k) - 1 \right|
```

为什么需要：Module C 需要知道每个 slot 应该看哪些 BB 指令，否则无法做可靠填槽。

## 5.8 同级顺序预测头

对同一 parent 下的子节点预测顺序，避免 CFG DFS 顺序和源码顺序不一致：

```math
r_i = W_{order} z_i
```

对 teacher 中应排在前面的 `(i, j)`：

```math
\mathcal{L}_{order}
= \sum_{(i,j)} \max(0, 1 - (r_j - r_i))
```

为什么需要：同级 if / loop / plain block 的输出顺序对骨架文本很关键。

---

# 6. 结构合法性约束

B_skel 不是完全自由生成文本，而是预测结构后做受约束解码。

## 6.1 约束项

```text
1. parent graph 必须是树或森林，最终挂到 FUNC root
2. IF 至少有 cond 和 then，可以没有 else
3. LOOP 至少有 cond 和 body
4. SWITCH 至少有 cond 和一个 case
5. 每个 BB 最多属于一个 primary slot
6. 每个非跳过 BB 必须被某个 slot 覆盖
7. slot 的 BB 集合尽量 CFG 连通
```

## 6.2 软约束损失

训练时可以加结构惩罚：

```math
\mathcal{L}_{valid}
= \lambda_{cyc}\mathcal{L}_{cycle}
+ \lambda_{cov}\mathcal{L}_{coverage}
+ \lambda_{role}\mathcal{L}_{role\_required}
+ \lambda_{conn}\mathcal{L}_{connectivity}
```

其中：

- `L_cycle` 惩罚 parent 形成环；
- `L_coverage` 惩罚 BB 漏覆盖或重复 primary owner；
- `L_role_required` 惩罚 IF/LOOP/SWITCH 缺少必要 role；
- `L_connectivity` 惩罚同一 slot 对应 BB 在 CFG 上过于分散。

## 6.3 推理时受约束解码

推理流程：

```text
1. 根据 p_ctrl 选择 IF/LOOP/SWITCH anchors
2. 用 parent scores 解码一棵控制结构树
3. 用 role scores 决定 cond/then/else/body/case/follow
4. 用 boundary + bb2slot 合并 BB 为 slot
5. 做合法性修复：去环、补缺失 role、去重 owner、排序
6. 序列化为 skeleton_text + slot metadata
```

这里规则不再负责“凭空恢复全部结构”，只负责把模型预测转成合法 CST。

---

# 7. B_align 的新定位

## 7.1 是否还需要 B_align

如果训练集已经有可靠 `M_gt`，那么 B_align 不是构造 teacher 的必要条件。它不能作为推理主路径，因为推理时没有源码。

但是它仍然可以作为辅助分支：

1. 帮助 `H_b` 学到更接近源码结构的表示；
2. 给论文保留“跨图结构对齐”的贡献点；
3. 提供 Node Alignment F1 / Region Alignment F1 等可解释指标；
4. 未来如果有部分数据没有 `M_gt`，可以用 B_align 做弱标注扩展。

## 7.2 B_align 输入输出

```text
输入：H_b, P_pat, E_b, H_s, T_s, E_s
输出：M_pred, H_hat_b, H_hat_s
```

注意：

```text
B_align 只在训练期和分析期运行。
推理期主流程不读取 M_pred。
```

## 7.3 B_align 损失保留为辅助项

原来的对齐损失可以保留：

```math
\mathcal{L}_{align}
= \mathcal{L}_{SCOT}
+ \lambda_{edge}\mathcal{L}_{edge}
+ \lambda_{region}\mathcal{L}_{region}
+ \lambda_{ca}\mathcal{L}_{ca\_align}
```

其中：

```math
\mathcal{L}_{SCOT}
= \text{KL}(M^{gt} || M^{pred})
```

```math
\mathcal{L}_{edge}
= \text{BCE}((M^{pred})^T A_b M^{pred}, A_s)
```

```math
\mathcal{L}_{ca\_align}
= \text{KL}(M^{gt} || A_{cross})
```

最终 Module B 总损失为：

```math
\mathcal{L}_B
= \mathcal{L}_{skel}
+ \lambda_{align}\mathcal{L}_{align}
```

如果短期只想把算法救起来，可以先设：

```math
\lambda_{align}=0
```

先把 `B_skel` 训练稳定，再逐步加辅助对齐。

---

# 8. B_skel 完整损失函数

## 8.1 主损失

```math
\mathcal{L}_{skel}
= \alpha_1 \mathcal{L}_{ctrl}
+ \alpha_2 \mathcal{L}_{parent}
+ \alpha_3 \mathcal{L}_{role}
+ \alpha_4 \mathcal{L}_{bd}
+ \alpha_5 \mathcal{L}_{bb2slot}
+ \alpha_6 \mathcal{L}_{order}
+ \alpha_7 \mathcal{L}_{valid}
```

推荐初始权重：

```text
alpha_ctrl    = 1.0
alpha_parent  = 1.0
alpha_role    = 1.0
alpha_bd      = 0.5
alpha_bb2slot = 1.0
alpha_order   = 0.2
alpha_valid   = 0.1
```

## 8.2 类别不均衡处理

IF/LOOP/SWITCH 的数量比普通 BB 少，switch_case 更少，需要类别权重：

```math
\mathcal{L}_{ctrl}
= - \sum_i w_{y_i} \log p_i^{ctrl}(y_i)
```

```math
w_c = \frac{1}{\sqrt{\text{freq}(c)+\epsilon}}
```

## 8.3 Module A 的 pattern loss 是否还保留

保留，但不要把它当最终目标。

```math
\mathcal{L}_{A}
= \mathcal{L}_{pattern\_cls}
```

联合微调时：

```math
\mathcal{L}_{A+B}
= \mathcal{L}_{skel}
+ \lambda_{pat}\mathcal{L}_{pattern\_cls}
+ \lambda_{align}\mathcal{L}_{align}
```

原因：Module A 的 `if_header/loop_header` 很有用，但 `then/else body` 和 `switch_case_body` 可能不稳定。B_skel 应该使用 `P_pat` 的 soft distribution，而不是完全相信 argmax。

---

# 9. Module C 接口修改

## 9.1 原接口问题

原接口是：

```text
Skeleton + H_hat_b + M -> Module C -> full source
```

问题是推理期没有源码侧 `M`，而且如果 skeleton 是 canonical CST，直接用原始完整源码做 teacher 会和骨架形态冲突。

## 9.2 新接口

Module C 输入改为：

```text
skeleton_text
slot_list
slot_to_BB
per_slot_binary_context
optional graph KV from H_b / z_b
```

示例：

```json
{
  "skeleton": "if (<FILL_COND:slot_1>) { <FILL_BLOCK:slot_2> }",
  "slots": [
    {
      "slot_id": "slot_1",
      "slot_type": "FILL_COND",
      "bb_indices": [3, 4],
      "binary_context": "cmp ...; jne ..."
    },
    {
      "slot_id": "slot_2",
      "slot_type": "FILL_BLOCK",
      "bb_indices": [5, 6, 7],
      "binary_context": "mov ...; call ..."
    }
  ]
}
```

## 9.3 Module C 训练目标

不要再把完整源码直接当作唯一 teacher，而是构造 slot-level teacher：

```json
{
  "slot_1": "n <= 0",
  "slot_2": "return NULL;",
  "slot_3": "buf[i++] = table[x];"
}
```

填槽损失：

```math
\mathcal{L}_{fill}
= - \sum_{k \in slots} \sum_{t \in y_k}
  w_k \log p(y_{k,t} | skeleton, slot_k, bin\_ctx_k)
```

如果还需要输出完整代码，则最终代码由：

```text
canonical skeleton + filled slots -> rendered source
```

得到。

---

# 10. Module D / RL 修改

## 10.1 原结构奖励问题

如果 `CST_bin` 来自规则 builder，而规则本身不稳定，那么 RL 的结构奖励也会不稳定。

## 10.2 新结构奖励

用 B_skel 输出作为 canonical binary CST：

```math
R_{struct}
= \text{sim}(\text{CST}_{gen}, \text{CST}_{B\_skel})
```

其中 `CST_gen` 是把生成代码 parse 后规范化得到的 CST。

建议默认：

```text
冻结 Module A
冻结 B_skel
只更新 Module C
```

等 Module C 稳定后，再考虑小学习率联合微调 B_skel。

---

# 11. 训练流程

## 11.1 Stage 0：Teacher 构造与审计

目标：离线生成 B_skel 的监督数据。

输入：

```text
训练 JSONL
source AST / source CFG
M_gt / alignment_matrix
binary CFG
```

输出：

```text
cst_teacher
slot_teacher
bb_to_slot labels
parent labels
role labels
teacher_quality flags
```

需要实现：

```text
scripts/export_bskel_teacher_from_ast_mgt.py
struclift/data/cst_teacher.py
```

审计指标：

```text
teacher_parse_ok
source_bounds_ok
M_gt coverage
CST node count
slot coverage
BB missing / duplicated
unrecoverable ratio
```

## 11.2 Stage 1：Module A 训练或复用

目标：得到稳定的 `H_b` 和 `P_pat`。

保留现有训练：

```math
\mathcal{L}_1
= \mathcal{L}_{InfoNCE}
+ \lambda_{pat}\mathcal{L}_{pattern\_cls}
```

建议：

```text
1. 继续保留 pattern label 训练
2. 重点关注 header 类：if_header / loop_header / switch_header
3. body 类标签作为软提示，不作为最终骨架判断的硬规则
```

## 11.3 Stage 2a：训练 B_skel 主分支

输入：

```text
binary CFG
H_b
P_pat
cst_teacher labels
```

冻结：

```text
Module A 初期冻结
```

训练：

```math
\mathcal{L}_{2a}
= \mathcal{L}_{skel}
```

先训简单任务：

```text
epoch 1-3: ctrl + parent
epoch 4-6: ctrl + parent + role
epoch 7+:  ctrl + parent + role + boundary + bb2slot + order + valid
```

这样比一次性训练所有头更稳。

## 11.4 Stage 2b：可选加入 B_align 辅助

如果想保留论文里的跨图对齐贡献：

```math
\mathcal{L}_{2b}
= \mathcal{L}_{skel}
+ \lambda_{align}\mathcal{L}_{align}
```

建议：

```text
先 lambda_align = 0
B_skel 稳定后 lambda_align = 0.05 或 0.1
不要让 B_align 的 M_pred 进入推理主链路
```

## 11.5 Stage 2c：可选联合微调 Module A + B_skel

当 B_skel 已经稳定后，小学习率解冻 Module A：

```math
\mathcal{L}_{2c}
= \mathcal{L}_{skel}
+ \lambda_{pat}\mathcal{L}_{pattern\_cls}
```

建议：

```text
lr_A = 0.1 * lr_B
只微调后几层 encoder 和 pattern classifier
```

目标是让 Module A 的表示服务于最终 CST，而不是只优化局部 pattern accuracy。

## 11.6 Stage 3：Module C slot-level SFT

输入来自 B_skel：

```text
skeleton_text
slot_to_BB
per-slot binary context
```

teacher 来自 canonical CST slot：

```text
slot_id -> source text span
```

损失：

```math
\mathcal{L}_3
= \mathcal{L}_{fill}
+ \lambda_{compile}\mathcal{L}_{compile}
```

训练策略：

```text
1. 前期用 teacher skeleton 训练 C，避免 B_skel 噪声拖垮 C
2. 中期混入 B_skel predicted skeleton，做 scheduled sampling
3. 后期用 predicted skeleton 为主，贴近推理
```

## 11.7 Stage 4：结构一致性 RL

目标：让填槽结果更可编译、更接近 B_skel 的 canonical CST。

奖励：

```math
R
= \beta_1 R_{compile}
+ \beta_2 R_{test}
+ \beta_3 R_{struct}
+ \beta_4 R_{slot}
```

其中：

```math
R_{struct}
= \text{TEDSim}(\text{CST}_{gen}, \text{CST}_{B\_skel})
```

默认只更新 Module C：

```text
Module A freeze
B_skel freeze
Module C update
```

---

# 12. 代码需要修改的位置

## 12.1 配置

文件：

```text
struclift/config.py
```

修改：

```text
ModuleBConfig 增加 B_skel 配置：
- skel_num_ctrl_types
- skel_num_roles
- skel_num_slot_types
- skel_gnn_layers
- skel_parent_topk
- lambda_ctrl
- lambda_parent
- lambda_role
- lambda_boundary
- lambda_bb2slot
- lambda_order
- lambda_valid
- lambda_align_aux
```

## 12.2 数据结构

文件：

```text
struclift/data/structures.py
```

增加：

```text
CSTNode
CSTSlot
SkeletonPrediction
SlotAssignment
```

`Skeleton` 需要扩展：

```text
slots: List[SlotInfo]
control_nodes: List[CSTNode]
slot_to_bb: Dict[str, List[int]]
source: "teacher" | "b_skel_pred"
quality: Dict[str, Any]
```

## 12.3 Teacher 构造

新增：

```text
struclift/data/cst_teacher.py
scripts/export_bskel_teacher_from_ast_mgt.py
scripts/audit_bskel_teacher_first100.py
```

功能：

```text
source AST -> canonical CST
M_gt -> BB-to-slot / BB-to-control projection
输出 stage2_bskel JSONL
```

## 12.4 Module B 模型

文件：

```text
struclift/models/module_b.py
```

新增：

```text
BinarySkeletonPredictor
SkeletonPredictionHead
ParentPointerHead
RoleHead
SlotBoundaryHead
BBToSlotHead
```

现有：

```text
CrossGraphAlignmentModule
```

保留，但改成辅助分支或兼容旧 checkpoint 的子模块。

## 12.5 模型总装

文件：

```text
struclift/models/struclift.py
```

修改：

```text
StrucLiftEncoderOutput 增加：
- skeleton_prediction
- skeleton_list from B_skel
- slot_to_bb_list

encode() 增加：
- run_b_skel=True
- run_alignment_aux=False
```

## 12.6 Stage 2 训练器

文件：

```text
struclift/training/stage2_alignment.py
```

建议改名或新增：

```text
struclift/training/stage2_skeleton.py
```

训练目标改为：

```text
主：B_skel losses
辅：B_align losses optional
```

## 12.7 Collator / Dataset

文件：

```text
struclift/data/binskel_jsonl.py
```

增加 batch 字段：

```text
cst_ctrl_labels
cst_parent_labels
cst_role_labels
cst_boundary_labels
cst_slot_labels
cst_order_pairs
cst_teacher_weights
```

## 12.8 Skeleton 序列化

文件：

```text
struclift/utils/skeleton.py
```

修改方向：

```text
旧：pattern labels + CFG rule -> skeleton
新：SkeletonPrediction -> canonical Skeleton
```

规则 builder 保留为 fallback / baseline / debug，不再作为主算法。

## 12.9 Stage 3 数据构造

相关脚本：

```text
scripts/augment_binskel_sft*.py
scripts/train_stage3_binskel_autoconfig.py
```

修改：

```text
使用 B_skel skeleton + slot_to_BB
构造 slot-level teacher
不要把原始 full source 直接当作 skeleton 约束下的唯一 GT
```

## 12.10 评估脚本

新增或修改：

```text
scripts/eval_bskel_cst_accuracy.py
scripts/export_bskel_pred_first100.py
scripts/audit_bskel_slot_mapping.py
```

指标：

```text
control node exact
parent exact
role exact
slot coverage
BB-to-slot accuracy
canonical skeleton exact
tree edit distance
semantic-equivalent skeleton rate
unrecoverable-filtered exact
```

---

# 13. 推荐最小落地顺序

为了避免一次改太大，建议分四步。

## Step 1：先做 teacher exporter

产物：

```text
stage2_bskel_teacher_train.jsonl
stage2_bskel_teacher_val.jsonl
```

先人工审计 100 条：

```text
source AST canonical CST 是否合理
slot 是否是一段区域而不是一 BB 一句
BB-to-slot 是否覆盖完整
```

只有 teacher 质量过关，后面训练才有意义。

## Step 2：只训练 ctrl + parent + role

先不做复杂 slot 生成，只预测控制树：

```math
\mathcal{L}
= \mathcal{L}_{ctrl}
+ \mathcal{L}_{parent}
+ \mathcal{L}_{role}
```

目标：

```text
control count / nesting exact 明显超过 rule builder
```

## Step 3：加入 slot boundary + BB-to-slot

再训练：

```math
\mathcal{L}
= \mathcal{L}_{ctrl}
+ \mathcal{L}_{parent}
+ \mathcal{L}_{role}
+ \mathcal{L}_{bd}
+ \mathcal{L}_{bb2slot}
```

目标：

```text
slot coverage 高
BB missing / duplicated 低
每个 slot 能聚合多个 BB
```

## Step 4：接 Module C

用 teacher skeleton 先训练 C，再逐步切到 predicted skeleton：

```text
epoch 1-2: 100% teacher skeleton
epoch 3-4: 50% teacher + 50% predicted
epoch 5+:  predicted skeleton
```

---

# 14. 为什么这个方案比原方案可信

## 14.1 推理闭环

修复前：

```text
训练依赖源码 M，推理没有源码 M
```

修复后：

```text
训练用源码生成 label，推理只用二进制预测 label
```

这是闭环的。

## 14.2 直接优化最终目标

修复前 Module B 的 loss 主要优化：

```text
BB <-> source 对齐
edge recovery
region contrastive
cross-attention alignment
```

但这些不是最终骨架质量。

修复后 Module B 直接优化：

```text
control type
parent/nesting
branch role
slot boundary
BB-to-slot
canonical skeleton
```

这和目标一致。

## 14.3 兼容原文贡献

不是完全抛弃原文，而是把原文改成：

```text
跨图对齐用于训练期 teacher / auxiliary representation learning；
二进制侧骨架预测用于推理期主流程。
```

这样仍然保留“结构学习”和“跨图对齐”的贡献，但不会让推理依赖不存在的源码。

---

# 15. 风险和边界

## 15.1 不是所有源码结构都能从二进制恢复

例如：

```text
编译器完全消除的 if
O2/O3 下合并的循环
switch lowering 成 if-chain
宏展开导致的源码结构
```

这些应该标为 `unrecoverable` 或只计入 semantic-equivalent 指标，不应该强压 exact AST。

## 15.2 Teacher 会有噪声

`M_gt`、source bounds、AST parse 都可能错。所以训练必须有：

```text
teacher_quality
loss weight
unreliable filtering
manual audit
```

## 15.3 Canonical CST 必须固定

如果 canonical 规则一直变，Module C 的 teacher 也会变，训练会混乱。必须先冻结一版 canonical CST 标准，再开始大规模训练。

---

# 16. 最终修改后的 Stage 总览

```text
Stage 0  Teacher construction
         source AST + M_gt -> canonical CST / slot labels

Stage 1  Module A
         binary CFG encoder + pattern classifier

Stage 2  Module B_skel
         binary-only CST / slot predictor
         optional B_align auxiliary

Stage 3  Module C
         slot-level SFT with skeleton + slot_to_BB

Stage 4  Module D
         structure consistency RL, default update C only
```

可以写成公式：

```math
\text{Teacher}: \quad
(AST_s, M^{gt}, G_b) \rightarrow Y^{CST}
```

```math
\text{Inference}: \quad
(G_b, H_b, P_{pat}) \xrightarrow{B_{skel}} \hat{Y}^{CST}
\xrightarrow{\text{serialize}} Skeleton
```

```math
\text{Generation}: \quad
(Skeleton, slot\_to\_BB, H_b) \xrightarrow{C} \hat{Code}
```

---

# 17. 结论

当前最关键的修复不是继续手写更复杂的 CFG builder，而是把“骨架恢复”变成 Module B 的直接监督目标。

推荐最终定义：

```text
Module B = B_skel + optional B_align
```

其中：

```text
B_skel 是主线：二进制侧输入，预测 canonical CST / slots / BB-to-slot。
B_align 是辅助：训练期跨图对齐，不参与推理主链路。
```

这样算法才满足：

```text
训练时可以利用源码监督；
推理时不依赖源码；
Module B 训练目标直接对应骨架恢复；
Module C 的填槽输入和 teacher 一致。
```

