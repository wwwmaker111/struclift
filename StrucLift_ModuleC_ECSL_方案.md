# StrucLift 模块 C 槽位算法改进方案：ECSL

> 适用目标：在不修改模块 A 与模块 B 的前提下，使模块 C 能够接受模块 B 生成的“CFG 语义正确但与源码 AST/标签不一致”的代码骨架，并稳定完成槽位填充。
> 核心结论：模块 C 不再学习模仿源码 AST 上的槽位标签，而是学习在模块 B 固定骨架内满足可验证的语义契约。

## 1. 问题定义

当前 StrucLift 的总体流程可以概括为：

- 模块 A：从二进制中编码控制流图、基本块、数据流、类型与语义特征。
- 模块 B：进行多粒度跨图对齐，并生成带 `<FILL_*>` 槽位的结构化代码骨架。
- 模块 C：在模块 B 给出的骨架中填充槽位，生成接近源码语义的代码。
- 模块 D：通过结构一致性强化学习进一步优化整体生成结果。

现有模块 C 的隐含假设是：源码 AST 中的 teacher slot 可以直接投影到模块 B 的 skeleton slot 上。但当前问题表明，这个假设不再成立。模块 B 生成的骨架可能与源码 AST 不同，却仍然是语义正确的控制结构。例如：

- 源码是 early return，模块 B 骨架是 if-else。
- 源码是 while，模块 B 骨架是 loop + break。
- 源码是 switch，模块 B 骨架是 if-chain。
- 源码是 `if (!x) return`，模块 B 骨架是 `if (x) { ... } else return`。
- 源码嵌套层级与模块 B 骨架不同，但 CFG 边、基本块覆盖和可观察行为一致。

因此，继续使用“源码 AST 节点到模块 B 槽位”的直接投影，会把正确的 B 骨架误判为标签错误，导致模块 C 学到带系统噪声的监督信号。

本方案将模块 C 改造为：

**ECSL：Execution-validated Compatible Semantic Labeling，执行验证驱动的骨架兼容语义槽位学习。**

其训练标签不再是源码 AST 同构标签，而是针对模块 B 固定骨架合成出来的、经过语义验证的 skeleton-compatible slot labels。

## 2. 综合结论

本方案分别从反编译、程序语义、机器学习、编译器验证和系统工程角度评审该问题。

### 2.1 方面 A：反编译与 CFG 恢复

意见：

- 模块 B 的骨架只要满足 CFG 覆盖、边保真和基本控制语义，就不应因 AST 形态不同而被拒绝。
- 源码结构不是唯一正确答案，尤其在编译优化、反编译规约和结构化恢复阶段，多个结构化程序可以对应同一个底层 CFG。
- 模块 C 必须尊重模块 B 的结构决策，不应回头要求模块 B 与源码 AST 同构。

建议：

- 将模块 C 的输入从“源码 AST 标签”改为“B 骨架槽位 + 二进制语义区域 + 可观察行为契约”。
- 每个槽位必须绑定到 B 骨架中的 anchor、CFG region、入口出口、可支配/后支配边界和数据流边界。

### 2.2 方面 B：程序语义与等价验证

意见：

- CFG 等价并不必然意味着完整语义等价。模块 C 还需要验证数据流、内存读写、副作用顺序、返回值和分支决策。
- 槽位级别标签必须有可验证契约，不能只用文本相似度或 AST 编辑距离判断正确性。

建议：

- 定义 `SemanticSlot` 数据结构。
- 每个候选填充值必须经过多级 validator：解析/类型检查、CFG successor 检查、SSA live-out 检查、读写集合和副作用检查、可选符号执行或差分执行。

### 2.3 方面 C：结构化 LLM 与弱监督学习

意见：

- 当前标签噪声是结构性噪声，不是随机噪声。简单 label smoothing 或直接丢弃样本不能解决核心问题。
- 正确做法是为同一个槽位构造多参考语义标签，并使用置信度加权的 marginal likelihood 进行训练。

建议：

- 高置信候选进入 SFT。
- 中置信候选进入弱监督、EM 或 reranker 训练。
- 低置信候选只作为检索/候选先验，不进入硬交叉熵。

### 2.4 方面 D：编译器与结构重写

意见：

- early return 与 if-else、while 与 loop + break、switch 与 if-chain 等差异是典型结构规约差异。
- 这些差异不能通过字符串匹配修复，应通过显式 rewrite candidate generator 生成等价候选，再交给 verifier 判定。

建议：

- 增加一组结构不一致专用候选生成器，但这些生成器只产生槽位候选，不修改模块 B 骨架。
- 所有 rewrite 必须有 guard condition，例如条件表达式是否无副作用、break/continue 目标是否一致、switch fallthrough 是否保留。

### 2.5 方面 E：系统工程与 SOTA 评审

意见：

- 算法必须可落地、可审计、可 ablation，不能只停留在“语义等价”概念层。
- 如果要支撑 SOTA 论文，必须构建 AST-nonisomorphic but CFG-equivalent 的专门测试集，并报告槽位契约满足率、CFG Edge F1、执行正确率、编译率等指标。

建议：

- 增加 B 骨架可靠性 gate。B 明显错误时，C 不应被迫学习错误骨架。
- 设计 curriculum：先训练 AST 基本同构样本，再逐步引入 branch flip、early return、loop rewrite、switch rewrite、O2/O3 复杂结构。

### 2.6 综合评审结论

最终形成一致意见：

1. 不修改模块 A/B 是可行的。
2. 需要把模块 C 的监督信号从 AST-isomorphic labels 改成 skeleton-compatible semantic labels。
3. 需要将模块 B 的 skeleton 视为固定合法控制结构，模块 C 的动作空间只允许填充槽位。
4. 需要用 binary/CFG/SSA/PDG 作为语义 pivot，而不是用源码 AST 作为唯一 teacher。
5. 需要通过验证器和置信度机制过滤噪声标签。
6. 该方案具备成为 SOTA 的潜力，因为它正面解决了反编译任务中“一对多结构化源码恢复”的核心问题。

## 3. 最优方案总览

最终建议将模块 C 改造成以下流水线：

```text
输入：
  binary-derived CFG/SSA/PDG from Module A
  skeleton with slots from Module B
  optional source teacher during training

步骤：
  1. Skeleton Slot Lowerer
     将 B 骨架降低到带槽位锚点的 skeleton CFG。

  2. Semantic Contract Builder
     为每个槽位构造 SemanticSlot 契约，包括控制流、数据流、类型、副作用和语法约束。

  3. SCTS Candidate Builder
     基于源码、二进制 IR、执行轨迹、结构重写模板和 LLM proposal 生成候选标签。

  4. Semantic Matcher
     使用约束匹配与 unbalanced optimal transport 找到源码语义片段与 B 槽位之间的兼容关系。

  5. Multi-level Validator
     对候选填充值进行解析、编译、CFG、SSA、效果和执行级验证，输出置信度。

  6. ECSL Training
     使用置信度加权多参考 SFT、EM、consistency training、自训练和 GRPO 训练模块 C。

  7. Global Slot Decoding
     推理阶段不独立贪心填每个槽位，而是生成 slot candidates 后做全局 beam/ILP/rerank。

输出：
  fixed B skeleton + verified slot fills
```

其中：

- **SCTS**：Skeleton-Compatible Teacher Synthesis，骨架兼容 teacher 合成。
- **ECSL**：Execution-validated Compatible Semantic Labeling，执行验证驱动的兼容语义槽位学习。

论文中的一句核心表述可以写为：

> ECSL converts AST-isomorphic slot supervision into verifier-backed, skeleton-compatible semantic slot supervision, enabling Module C to train on CFG-equivalent but AST-divergent skeletons without modifying Modules A/B.

## 4. 核心数据结构：SemanticSlot

模块 C 的基本监督单元不再是 AST 节点，而是 `SemanticSlot`。

推荐 schema：

```yaml
SemanticSlot:
  slot_id: string
  slot_type:
    enum:
      - COND
      - LOOP_EXIT
      - SWITCH_DISPATCH
      - CASE_LABEL
      - EXPR
      - LVALUE
      - RETURN
      - CALL_ARG
      - CALL_TARGET
      - MEMORY_EFFECT
      - PHI_OR_MERGE
      - STMT
      - BLOCK
      - EXIT

  anchor:
    skeleton_ast_path: string
    dfs_order: int
    parent_construct: string
    placeholder_text: string

  region:
    entry_bb: string
    exit_bbs: list[string]
    region_bbs: list[string]
    covered_edges: list[tuple[string, string, string]]
    binary_inst_span: optional[string]
    source_span: optional[string]

  control_contract:
    true_successor: optional[string]
    false_successor: optional[string]
    fallthrough_successor: optional[string]
    postdom_successor: optional[string]
    loop_header: optional[string]
    loop_latch: optional[string]
    break_target: optional[string]
    continue_target: optional[string]
    allowed_control_effects:
      - none
      - return
      - break
      - continue
      - goto_like_internal

  data_contract:
    live_in: list[string]
    live_out: list[string]
    defs: list[string]
    uses: list[string]
    type_hints: map[string, string]
    canonical_symbols: map[string, string]

  effect_contract:
    read_set: list[string]
    write_set: list[string]
    call_effects: list[string]
    volatile_ops: list[string]
    side_effect_order: list[string]
    may_throw_or_longjmp: bool

  grammar_contract:
    category:
      enum:
        - expression_only
        - condition_only
        - statement_list
        - compound_block
        - return_expr
        - lvalue_expr
        - call_target
    allowed_tokens: optional[list[string]]
    forbidden_control_transfer: list[string]
    scope_constraints: list[string]

  polarity:
    enum:
      - normal
      - inverted
      - unknown

  validation_rule:
    local_stub_signature: string
    compile_context: string
    local_equivalence_observation: list[string]

  confidence:
    initial_alignment_score: float
    validation_score: float
    final_weight: float
```

### 4.1 可观察语义 Obs(slot)

槽位等价不比较 AST 形状，而比较可观察语义：

```text
Obs(slot) =
  branch successor decision,
  live-out variable values,
  return value if affected,
  memory write footprint,
  externally visible call effects,
  volatile operation sequence,
  required control transfer target.
```

不应比较：

- 临时变量名字。
- AST 节点类型是否一致。
- 非 live-out 的中间值。
- 语句是否拆分或合并。
- early return 与 if-else 这类结构表面差异。

## 5. 统一 IR 与语义 lowering

为了避免源码 AST 与 B 骨架直接对齐，训练阶段应将三类对象降低到统一语义 IR：

- source teacher code；
- module B skeleton；
- binary-derived representation from module A。

推荐 IR node：

```yaml
IRNode:
  id: string
  kind:
    enum:
      - cond
      - assign
      - call
      - return
      - memory_read
      - memory_write
      - phi
      - switch
      - jump
      - arithmetic
      - compare
  bb_id: string
  operands: list[string]
  defs: list[string]
  uses: list[string]
  effects: list[string]
  type: optional[string]
  source_span: optional[string]
  binary_span: optional[string]
```

推荐 IR edge：

```yaml
IREdge:
  src: string
  dst: string
  edge_type:
    enum:
      - true
      - false
      - fallthrough
      - back
      - call
      - return
      - exceptional
```

变量命名使用 canonical symbol，而不是源码名字：

```text
arg_0, arg_1, stack_slot_0, stack_slot_1,
global_addr_0, ret_value, tmp_0, mem_region_0
```

这样可以减少源码命名差异和反编译命名差异带来的标签噪声。

## 6. Skeleton Slot Lowerer

该组件负责把模块 B 输出的 skeleton 降低为带槽位锚点的 skeleton CFG。

输入：

```text
B_skeleton:
  code tokens with <FILL_0>, <FILL_1>, ...
  structured constructs: if, else, loop, switch-like chain, return, call
  B validator metadata: BB coverage, edge preservation, alignment confidence
```

输出：

```text
SkeletonCFG:
  nodes: skeleton basic blocks
  edges: true/false/fallthrough/back
  slots: SemanticSlot[]
```

关键要求：

1. 每个 `<FILL_*>` 必须有明确 `slot_type`。
2. 每个槽位必须绑定到 skeleton AST path 和 CFG region。
3. 条件槽位必须记录 true/false successor。
4. 循环槽位必须记录 header、latch、break target、continue target。
5. switch/if-chain 槽位必须记录 dispatch variable、case label、default successor。
6. statement/block 槽位必须记录 live-in、live-out、read/write/effect。
7. lowerer 不得修改 B 骨架，只能添加元数据。

## 7. SCTS：骨架兼容 Teacher 合成

SCTS 的目标是为每个 B skeleton slot 生成候选标签集合：

```text
Candidates(slot_i) = {
  candidate_1, candidate_2, ..., candidate_k
}
```

每个 candidate 不只是文本，而应有三层表示：

```yaml
SlotCandidate:
  candidate_id: string
  slot_id: string

  semantic_object:
    kind:
      - predicate
      - expression
      - lvalue
      - statement_transformer
      - return_value
      - memory_effect
    ir_refs: list[string]
    summary: string

  typed_canonical_ir:
    expr_tree: optional[object]
    stmt_region: optional[list[IRNode]]
    transformer:
      live_in_to_live_out: optional[string]
    type: optional[string]

  c_renderings:
    - text: string
      style: canonical
      requires: list[string]

  provenance:
    source:
      - source_cfg
      - binary_ir
      - execution_trace
      - rewrite_template
      - llm_proposal
      - self_training
    source_refs: list[string]

  validation:
    parse_ok: bool
    compile_ok: bool
    successor_match: bool
    live_out_match: bool
    effect_match: bool
    symbolic_or_trace_match: optional[bool]
    confidence_level: string
    confidence_weight: float
```

候选来源：

1. 源码 CFG/SSA 中与槽位 region 语义匹配的表达式或语句片段。
2. 二进制 lifted IR 中的条件、赋值、返回、内存读写和调用效果。
3. 动态执行轨迹或单元测试轨迹。
4. 显式结构 rewrite 模板。
5. LLM 根据 slot contract 生成的 proposal。
6. 通过高置信预测结果积累的 self-training candidate。

LLM proposal 只能作为候选来源，不能直接作为强标签。所有 proposal 必须经过 validator。

## 8. 显式结构不一致 Rewrite Candidate Generator

这些 generator 的作用是构造候选标签，不改变模块 B 骨架。

### 8.1 early return ↔ if-else

场景：

```c
if (!cond) return r;
body;
```

与：

```c
if (cond) {
  body;
} else {
  return r;
}
```

是可能等价的。

生成候选：

- `cond`
- `!cond`
- `return r`
- guarded body 对应的 statement/block fill

guard condition：

- `cond` 无副作用，或副作用顺序在两个结构中一致。
- `return r` 的返回值与 binary ret_value 一致。
- early exit 的 successor 与 B skeleton 的 else/exit successor 一致。
- body 的 live-in/live-out 与原 region 一致。

### 8.2 while ↔ loop + break

场景：

```c
while (cond) {
  body;
}
```

与：

```c
while (1) {
  if (!cond) break;
  body;
}
```

是可能等价的。

生成候选：

- loop guard 条件：`cond`
- break 条件：`!cond`
- loop body slot fills

guard condition：

- loop header、latch、exit target 一致。
- condition 的求值时机一致。
- continue target 未被改变。
- loop-carried variables 的 phi/live-out 一致。
- do-while 与 while 不得混淆。
- condition 含函数调用、volatile read 或内存写时，不允许随意重排。

### 8.3 switch ↔ if-chain

场景：

```c
switch (x) {
  case 1: ...
  case 2: ...
  default: ...
}
```

与：

```c
if (x == 1) { ... }
else if (x == 2) { ... }
else { ... }
```

是可能等价的。

生成候选：

- dispatch expression：`x`
- case condition：`x == k`
- default condition。

guard condition：

- case value 类型、符号扩展和整数宽度一致。
- fallthrough 行为被显式保留或确认不存在。
- default successor 一致。
- case block 的 read/write/effect 一致。

### 8.4 branch inversion / then-else swap

场景：

```c
if (!x) A; else B;
```

与：

```c
if (x) B; else A;
```

是可能等价的。

生成候选：

- `x`
- `!x`
- swapped branch mapping。

guard condition：

- true_successor 和 false_successor 必须匹配 B skeleton。
- condition 无副作用，或反转不改变副作用次数与顺序。
- branch body 对应 region 的 live-out 一致。

### 8.5 nested if ↔ conjunctive condition

场景：

```c
if (a) {
  if (b) {
    body;
  }
}
```

与：

```c
if (a && b) {
  body;
}
```

是可能等价的。

guard condition：

- 短路语义一致。
- `a` 为 false 时不得求值 `b`。
- `a` 与 `b` 的副作用必须与原结构一致。
- false successor 与后支配边界一致。

### 8.6 break/continue restructuring ↔ guarded body

场景：

```c
for (...) {
  if (skip) continue;
  body;
}
```

与：

```c
for (...) {
  if (!skip) {
    body;
  }
}
```

是可能等价的。

guard condition：

- continue target 与 loop latch 一致。
- body 内部的 break/continue 不得被错误吸收。
- loop-carried variables 与内存效果一致。

## 9. 语义匹配与 Unbalanced Optimal Transport

源码 teacher slice 与 B skeleton slot 之间可能是一对一、一对多、多对一或部分未匹配关系。因此建议使用约束匹配 + unbalanced optimal transport 作为 candidate search，不作为最终真值。

### 9.1 Cost Matrix

对 source semantic unit `u` 与 skeleton slot `s` 定义成本：

```text
Cost(u, s) =
  w_type      * type_mismatch
+ w_livein    * live_in_mismatch
+ w_liveout   * live_out_mismatch
+ w_rw        * read_write_mismatch
+ w_effect    * side_effect_mismatch
+ w_succ      * cfg_successor_mismatch
+ w_dom       * dominance_postdominance_mismatch
+ w_pdg       * pdg_dependency_mismatch
+ w_sem       * semantic_summary_distance
+ w_polarity  * branch_polarity_mismatch
+ w_edit      * rewrite_cost
- w_rewrite   * rewrite_compatibility_bonus
- w_verify    * verifier_confidence_bonus
- w_bb        * bb_overlap_bonus
```

注意：这个 `Cost(u, s)` 不是模块 C 训练 LLM 时直接反向传播的 loss。它是训练数据构造阶段的语义匹配成本，用于回答：

```text
源码/二进制中的哪个 semantic unit 可以作为当前 skeleton slot 的候选 teacher？
这个候选是否应进入 strong label、weak label，还是直接丢弃？
```

因此它的角色更接近 data labeling / candidate retrieval / alignment scoring，而不是 neural training objective。  
模块 C 真正训练时使用的是第 13 节的 confidence-weighted multi-reference SFT：

```text
L_sft(i) = - log sum_j c_ij * P_theta(y_ij | x_i, slot_i)
```

其中 `c_ij` 来自 matcher、validator 和 confidence scorer。也就是说，`Cost(u, s)` 只影响“哪些候选标签进入训练、每个候选标签权重多大”，不会直接要求模型优化这些不可微的静态特征。

专家认可该 cost 的原因不是因为它本身是一个好训的神经 loss，而是因为它把原本错误的一对一 AST 投影，改造成了可解释、可约束、可验证的候选标签生成过程。最终可训练性由以下机制保证：

- 训练目标仍然是标准 token likelihood / marginal likelihood，LLM 容易训练。
- 多个语义等价候选共同作为 reference，避免强迫模型拟合唯一源码 AST 形态。
- 低置信候选不会进入 hard CE，降低系统性标签噪声。
- validator 提供置信度，使训练信号更接近真实可执行语义。
- global decoding 和 GRPO 在推理后期继续优化可执行正确性。

### 9.2 Hard Constraints

以下约束不满足时，不允许匹配为强 teacher：

- entry/exit successor 一致。
- post-dominator boundary 不冲突。
- SCC/loop membership 合法。
- live-out definition 能被槽位解释。
- call effect 和 memory effect 不丢失。
- branch polarity 与 B skeleton successor 一致，或可通过显式 inversion 解释。
- 类型兼容。
- 不引入未定义行为。
- 不改变 B skeleton 的控制结构 token。

### 9.3 Unmatched Mass

允许存在未匹配质量：

- unmatched source slice：源码中多出的结构表达，不需要映射到 B slot。
- unmatched B slot：找不到可靠 teacher，只作为弱标签或由模型生成后验证，不进入硬 SFT。

这样可以避免为了对齐而强行制造错误监督。

## 10. 候选验证与置信度

每个候选 label 必须经过多级 validator。

### 10.1 Validator Levels

```text
V0: Parser / type / local stub validation
    候选文本可解析，类型兼容，满足 slot grammar。

V1: CFG successor + skeleton preservation
    填入候选后，B skeleton 结构不变，条件槽位的 true/false successor 正确。

V2: SSA live-out + read/write/effect validation
    live-out、defs/uses、memory read/write、call effect 与 slot contract 一致。

V3: Symbolic / SMT / differential execution validation
    在可行范围内验证候选与 binary/source semantic slice 的可观察行为一致。
```

### 10.2 Confidence Levels

```text
C5: parse + compile + exact symbolic equivalence
    weight = 1.0

C4: parse + compile + bounded symbolic execution
    weight = 0.8

C3: parse + compile + differential execution tests
    weight = 0.5

C2: parse + compile + CFG/live-out compatible only
    weight = 0.2

C1: heuristic semantic match only
    weight = 0.0 for hard CE, can be used as prior/reranker feature

C0: invalid / unmatched
    discard
```

候选最终置信度：

```text
conf(candidate) =
  base_alignment_score
+ parse_bonus
+ compile_bonus
+ successor_match_bonus
+ live_out_match_bonus
+ effect_match_bonus
+ symbolic_or_trace_bonus
- ambiguity_penalty
- rewrite_complexity_penalty
- unsupported_behavior_penalty
```

原则：

- C3 及以上可进入强监督。
- C2 可进入弱监督或 consistency training。
- C1 只能作为候选先验。
- C0 直接丢弃。

## 11. 模块 C 输入输出格式

### 11.1 输入格式

推荐模块 C 的输入不再只是 skeleton 文本，而是：

```text
<global>
function_signature: int foo(int arg_0, char *arg_1)
architecture: x86_64
compiler_hint: clang/O2/unknown
symbol_table: ...
type_hints: ...
</global>

<skeleton>
int foo(int arg_0, char *arg_1) {
  if (<FILL_0:COND>) {
    <FILL_1:BLOCK>
  } else {
    return <FILL_2:RETURN>;
  }
}
</skeleton>

<slot_contracts>
slot FILL_0:
  type: COND
  true_successor: bb_3
  false_successor: bb_8
  live_in: [arg_0, stack_slot_0]
  read_set: [mem_region_0]
  allowed_effects: none

slot FILL_1:
  type: BLOCK
  region_bbs: [bb_3, bb_4, bb_5]
  live_out: [ret_value, stack_slot_1]
  write_set: [stack_slot_1]

slot FILL_2:
  type: RETURN
  live_in: [arg_0]
  expected_type: int
</slot_contracts>
```

### 11.2 输出格式

推荐让模块 C 输出结构化 slot patches，而不是整段自由代码：

```json
{
  "fills": [
    {
      "slot_id": "FILL_0",
      "text": "arg_0 > 0"
    },
    {
      "slot_id": "FILL_1",
      "text": "stack_slot_1 = arg_0 + 1;"
    },
    {
      "slot_id": "FILL_2",
      "text": "-1"
    }
  ]
}
```

再由 deterministic renderer 将这些填充值放回 B skeleton。

renderer 负责检查：

- slot grammar 是否满足；
- 类型是否兼容；
- 局部变量作用域是否可见；
- 是否出现非法 return/break/continue；
- 是否试图修改 skeleton token；
- 是否引入额外函数、全局状态或未声明变量。

## 12. Attention Mask 改造

原始模块 C 若只关注与 slot 对齐的基本块，容易在 AST 不一致场景下漏掉边界信息。建议使用三环 attention mask：

```text
M_slot =
  alpha * region_bb
+ beta  * boundary_bb
+ gamma * dataflow_bb
```

其中：

- `region_bb`：槽位直接覆盖的基本块。
- `boundary_bb`：槽位入口、出口、支配/后支配边界、true/false successor。
- `dataflow_bb`：定义 live-in 或使用 live-out 的相关基本块。

推荐权重：

| Slot Type       | region | boundary | dataflow |
| --------------- | -----: | -------: | -------: |
| COND            |   0.55 |     0.35 |     0.10 |
| STMT/BLOCK      |   0.50 |     0.15 |     0.35 |
| RETURN          |   0.45 |     0.10 |     0.45 |
| PHI_OR_MERGE    |   0.10 |     0.45 |     0.45 |
| LOOP_EXIT       |   0.45 |     0.40 |     0.15 |
| SWITCH_DISPATCH |   0.45 |     0.40 |     0.15 |

该 mask 让模型在填条件时看清 successor，在填语句时看清数据依赖，在填返回值时看清 ret_value 来源。

## 13. ECSL 训练目标

### 13.1 Confidence-weighted Multi-reference SFT

对一个 slot，有多个验证通过的候选：

```text
Y_i = {y_i1, y_i2, ..., y_ik}
Conf_i = {c_i1, c_i2, ..., c_ik}
```

训练损失：

```text
L_sft(i) = - log sum_j c_ij * P_theta(y_ij | x_i, slot_i)
```

而不是只选择一个“最像源码 AST”的标签。

### 13.2 Latent EM

可选地对中置信候选使用 EM：

```text
E-step:
  q_ij ∝ c_ij * P_theta(y_ij | x_i, slot_i)

M-step:
  L_em(i) = - sum_j q_ij * log P_theta(y_ij | x_i, slot_i)
```

工程注意事项：

- 先用高置信 SFT warmup。
- EM 初期使用较高 temperature，避免过早塌缩。
- 设置 entropy floor。
- 设置 confidence cap，避免一个错误高分候选支配训练。
- C1/C0 候选不得进入 EM hard update。

### 13.3 Consistency Training

对同一个语义 region 构造不同结构视图：

- branch inversion view；
- early-return view；
- if-else view；
- loop+break view；
- source-like view；
- skeleton-like view。

要求模型在不同视图下生成满足同一 `Obs(slot)` 的填充值。

### 13.4 Self-training

当模型在未标注或低置信样本上生成候选后：

1. 通过 renderer 填回 skeleton。
2. 运行 validator。
3. 通过 C3/C4/C5 的结果加入 candidate pool。
4. 低于 C3 的结果不进入强监督。

### 13.5 GRPO / RL 阶段

GRPO 只更新模块 C adapter/LoRA，不修改 A/B。

动作空间：

```text
action = slot text only
```

硬约束：

- parse_ok；
- compile_ok；
- no skeleton mutation；
- slot grammar valid；
- no illegal control transfer。

奖励：

```text
R =
  alpha * R_compile
+ beta  * R_cfg_successor
+ gamma * R_slot_contract
+ delta * R_semantic_execution
+ eta   * R_live_out_effect
+ rho   * R_readability
- lambda_kl * KL(policy || reference)
- penalty_ub * UB_or_type_violation
- penalty_side_effect * extra_side_effect
- penalty_mutation * skeleton_edit
```

推荐初始权重：

```text
alpha = 0.15
beta  = 0.20
gamma = 0.20
delta = 0.25
eta   = 0.15
rho   = 0.05
```

其中 `R_semantic_execution` 在有测试、trace 或 symbolic validation 时权重最高。

## 14. 推理阶段全局解码

模块 C 推理时不应独立选择每个槽位 top-1，因为多个槽位之间存在全局一致性。例如：

- 条件槽位与后续 return 槽位共享变量。
- switch dispatch 与 case label 必须一致。
- pointer base/index 表达式必须一致。
- 多个分支的返回值必须符合统一函数语义。
- SSA value 命名和 C 局部变量声明必须统一。

推荐流程：

```text
for slot in topological_order(SkeletonCFG):
    generate k candidates with grammar-constrained decoding
    validate each candidate locally

global_search:
    use beam / ILP / DP reranker
    enforce cross-slot consistency
    render full function
    run full-function validator
    select best verified program
```

全局 rerank features：

- compile success；
- skeleton preservation；
- CFG edge F1；
- slot contract satisfaction；
- live-out equivalence；
- read/write/effect equivalence；
- differential execution score；
- symbolic score；
- variable naming coherence；
- code readability；
- length penalty。

## 15. B 骨架可靠性 Gate

虽然本方案接受 B 的 AST 差异，但不接受明显错误的 B 骨架。训练与推理都应有 B gate：

```text
if B_validator_pass
   and BB_coverage >= threshold
   and edge_preservation >= threshold
   and SkeletonSlotLowerer_success
   and contract_consistency_pass:
       use ECSL path
else:
       downweight / exclude from SFT
       optional: RL-only or fallback baseline
```

建议记录：

- BB coverage；
- CFG edge preservation；
- irreducible CFG flag；
- lowering success；
- slot contract conflict count；
- B alignment confidence；
- unsupported construct flag。

明显失败的 B 骨架不能作为 C 的强监督训练样本，否则会污染模块 C。

## 16. 训练 Curriculum

建议按难度分层训练：

### Level 0：AST 与 skeleton 基本同构

目标：

- 建立基础 slot grammar、类型、表达式和语句填充能力。

样本：

- direct if；
- direct while；
- direct return；
- direct call；
- simple assignment。

### Level 1：branch flip / then-else swap

目标：

- 学会 branch polarity 与 successor 绑定，而不是机械模仿源码条件。

样本：

- `if (!x)` ↔ `if (x) else`；
- then/else 顺序交换。

### Level 2：early return ↔ if-else

目标：

- 学会在不同结构下保持 exit 行为和 live-out 一致。

样本：

- guard return；
- error handling return；
- nested early exit。

### Level 3：while ↔ loop + break / statement split-merge

目标：

- 学会 loop guard、break target、continue target、loop-carried variable。

样本：

- `while(cond)`；
- `while(1){if(!cond)break;}`；
- loop body split/merge。

### Level 4：switch ↔ if-chain / O2/O3 复杂结构

目标：

- 学会 dispatch consistency、case coverage、fallthrough、优化后 CFG 的复杂映射。

样本：

- switch lowering；
- jump table；
- nested condition；
- compiler O2/O3 restructuring。

## 17. 实验设计

必须构建专门测试集：

**AST-nonisomorphic but CFG-equivalent subset**

建议包含以下 bucket：

- early return ↔ if-else；
- while ↔ loop + break；
- switch ↔ if-chain；
- branch inversion；
- guard/nested condition；
- statement split/merge；
- single expression ↔ multi-BB computation；
- O2/O3 compiler restructuring；
- pointer/memory effect sensitive cases；
- return-value restructuring。

### 17.1 主要指标

```text
compile_rate
re_execution_accuracy
CFG_edge_F1
branch_successor_accuracy
slot_contract_satisfaction_rate
live_out_equivalence_rate
read_write_effect_equivalence_rate
candidate_validation_precision
high_confidence_label_coverage
pass@1 / pass@k
structure_normalized_tree_edit_distance
O2/O3 robustness
```

### 17.2 Ablation

必须对比：

- 原始 AST projection Module C；
- + SCTS candidate synthesis；
- + SemanticSlot contract；
- + branch polarity / successor constraint；
- + explicit rewrite generator；
- + multi-reference marginal likelihood；
- + confidence weighting；
- + validator rerank；
- + global slot decoding；
- + GRPO；
- 去掉 binary pivot；
- 去掉 UOT matching；
- 去掉 confidence weighting；
- 去掉每类 rewrite generator；
- 只用 compile-only rerank。

### 17.3 目标阈值

建议论文或内部里程碑采用：

```text
high_confidence_label_coverage >= 80%
candidate_validation_precision >= 95%
compile_rate >= 95%
CFG_edge_F1 on O2/O3 >= 0.90
branch_successor_accuracy 显著高于原始 C
AST-nonisomorphic subset re-execution accuracy 提升 >= 8-10 points
slot_contract_satisfaction_rate >= 90%
```

## 18. 工程实现路线

### Phase 1：数据结构与 Lowerer

交付：

- `SemanticSlot` schema。
- `SkeletonCFG` schema。
- `SkeletonSlotLowerer`。
- slot grammar checker。
- deterministic renderer。

验收：

- 所有 B skeleton 的 slot 都能定位到 skeleton AST path。
- 条件、循环、返回、block 槽位有明确 contract。
- renderer 能保证 skeleton token 不被修改。

### Phase 2：Candidate Builder

交付：

- source CFG/SSA candidate extractor。
- binary IR candidate extractor。
- rewrite candidate generators。
- LLM proposal generator。
- candidate provenance tracker。

验收：

- 对 early return、loop+break、switch/if-chain、branch inversion 样本均能生成候选。
- 每个候选都有 semantic object、canonical IR、C rendering。

### Phase 3：Validator 与 Confidence

交付：

- V0 parser/type validator。
- V1 skeleton/CFG successor validator。
- V2 live-out/read-write-effect validator。
- V3 trace/symbolic validator。
- confidence scorer。

验收：

- C3+ 候选 precision 达到 95% 以上。
- 错误候选不会进入硬 SFT。

### Phase 4：ECSL 训练

交付：

- confidence-weighted multi-reference SFT。
- EM training optional。
- consistency training。
- self-training loop。

验收：

- AST-nonisomorphic subset 指标明显优于原始 C。
- 模型不会把源码 AST 形态错误地强加给 B skeleton。

### Phase 5：全局解码与 GRPO

交付：

- per-slot k candidate decoding。
- global beam/ILP/rerank。
- full-function validation。
- GRPO reward。

验收：

- pass@1 与 pass@k 提升。
- full function compile/re-execution 指标提升。
- A/B 冻结时，C 仍可在 B 结构差异下稳定生成正确代码。

## 19. 关键伪代码

### 19.1 训练样本构建

```python
def build_ecsl_training_sample(binary_ir, source_code, b_skeleton):
    skeleton_cfg = lower_skeleton_with_slots(b_skeleton)
    if not pass_b_gate(binary_ir, skeleton_cfg):
        return weak_or_skip_sample()

    source_ir = lower_source_to_ir(source_code)
    slots = build_semantic_slots(skeleton_cfg, binary_ir)

    sample = []
    for slot in slots:
        raw_candidates = []
        raw_candidates += extract_from_source_ir(source_ir, slot)
        raw_candidates += extract_from_binary_ir(binary_ir, slot)
        raw_candidates += generate_rewrite_candidates(source_ir, skeleton_cfg, slot)
        raw_candidates += propose_with_llm(slot, skeleton_cfg, binary_ir)

        matched_candidates = semantic_match(raw_candidates, slot)
        verified_candidates = []

        for cand in matched_candidates:
            rendered = render_slot_candidate(b_skeleton, slot, cand)
            report = validate_candidate(rendered, slot, binary_ir)
            cand.confidence = score_confidence(cand, report)
            if cand.confidence.level >= "C2":
                verified_candidates.append(cand)

        sample.append({
            "slot": slot,
            "candidates": verified_candidates
        })

    return sample
```

### 19.2 多参考 SFT

```python
def ecsl_sft_loss(model, x, slot, candidates):
    valid = [c for c in candidates if c.confidence.weight > 0]
    if not valid:
        return None

    terms = []
    for c in valid:
        logp = model.log_prob(c.text, condition=(x, slot))
        terms.append(log(c.confidence.weight) + logp)

    return -logsumexp(terms)
```

### 19.3 推理

```python
def infer_with_ecsl(model, binary_ir, b_skeleton):
    skeleton_cfg = lower_skeleton_with_slots(b_skeleton)
    slots = topological_order(skeleton_cfg.slots)

    candidate_table = {}
    for slot in slots:
        prompt = build_slot_prompt(binary_ir, skeleton_cfg, slot)
        candidates = model.generate_k(prompt, k=8, grammar=slot.grammar_contract)
        candidates = local_validate_and_score(candidates, slot, binary_ir)
        candidate_table[slot.slot_id] = candidates

    programs = global_decode(b_skeleton, candidate_table)
    programs = full_function_validate(programs, binary_ir)

    return select_best(programs)
```

## 20. 风险与边界

主要风险：

1. **Teacher synthesis 噪声**解决：多级 validator、confidence weighting、低置信样本不进入 hard CE。
2. **CFG 等价不足以证明语义等价**解决：加入 live-out、read/write/effect、return value、trace/symbolic validation。
3. **局部 slot 等价不推出全局函数等价**解决：推理阶段做 full-function validation 和 global decoding。
4. **复杂 heap、I/O、异常、longjmp、并发语义难以验证**解决：标记 unsupported behavior，降低置信度，进入弱监督或跳过。
5. **loop invariant 和优化后复杂循环难验证**解决：bounded symbolic execution、差分测试、专用 loop templates、保守置信度。
6. **B 骨架本身错误**
   解决：B gate；明显错误的 skeleton 不作为强监督样本。

## 21. 最终可执行结论

在不改变模块 A 和 B 的前提下，模块 C 的最优修改不是继续修补 AST 投影规则，而是整体改为：

```text
固定 B skeleton
    ↓
为每个 slot 构造 SemanticSlot contract
    ↓
合成 skeleton-compatible teacher candidates
    ↓
通过 CFG/SSA/effect/execution validator 赋置信度
    ↓
使用多参考、置信度加权、结构一致性训练模块 C
    ↓
推理时进行全局 slot 解码与 full-function validation
```

这样，模块 C 学到的是：

```text
在给定 B 骨架约束下，生成满足二进制语义和槽位契约的代码片段。
```

而不是：

```text
复现源码 AST 的局部结构形态。
```

该方案能够直接解决“B 骨架语义正确但与 label 不一致”导致的 C 槽位监督噪声问题，并保留 A/B 模块的现有设计。若按照上述实现路线完成 lowerer、candidate synthesis、validator、confidence-weighted training 和 global decoding，整体系统有望在 AST-nonisomorphic but CFG-equivalent 场景下显著超过原始 Module C，并形成具备 SOTA 竞争力的研究贡献。
