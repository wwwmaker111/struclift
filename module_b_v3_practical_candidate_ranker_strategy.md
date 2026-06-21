# Module B-v3：候选生成 + 排序 + 验证的可落地算法实施方案

核心变化：不再训练模型直接生成 skeleton；改为规则生成合法候选，模型只排序，validator 强制兜底。

---

## 0. 一句话结论

Module B-v3 应改成：

```text
Binary CFG/SSA
  -> Deterministic Normalization and Analysis
  -> Candidate Generator
  -> Small Candidate Ranker
  -> Validator / Canonicalizer / Fallback
  -> Verified Canonical Structured Skeleton
```

核心原则：

1. **正确性不靠模型学习**：CFG 覆盖、边守恒、SSA 合法性、grammar 合法性必须由确定性算法和 validator 保证。
2. **模型不直接生成最终 skeleton**：模型只在一组已经合法或可 fallback 的候选中排序。
3. **每个 region 必须有 fallback**：任何结构化失败的区域都能退化为 `UNSTRUCTURED_REGION` 或 `GOTO_SLOT`。
4. **第一版只训练一个主 loss**：`L_rank`。slot grounding 可作为第二阶段单独加入。
5. **系统先追求不出错，再追求漂亮**：accepted cases 必须 CFG 等价；无法处理时显式拒绝或 fallback。

---

## 1. 基于给出的原 loss 方案难落地的问题

原方案中包含：

```text
L_match
L_ctrl
L_region
L_parent
L_role
L_slot
L_ground
L_uot
L_prod
L_cons
L_cfg
L_rank
```

这些 loss 的问题不是“数量多”，而是目标边界错了。

### 1.1 硬正确性被误写成可学习目标

以下目标不应让模型通过 gradient 学：

- CFG edge preservation。
- BB / instruction coverage。
- parent tree 无环。
- grammar production 合法。
- region containment。
- SSA def-use 合法。
- condition polarity 与 true/false edge 一致。
- side-effect ordering。

这些目标错一次就会导致反编译结果不可信，因此必须由规则、mask、validator 和 fallback 处理。

### 1.2 标签非唯一

同一个 binary CFG 可能有多个合法 skeleton：

```text
if (c) { A } else { B }
if (!c) { B } else { A }
guard if + goto
loop + break
structured region + residual goto
```

如果强行用 exact-match tree loss、parent loss、role loss，会把合法等价解当成错误，训练会不稳定。

### 1.3 Grounding 不应主要靠模型猜

instruction/SSA-level grounding 本质是 provenance 问题：

```text
instruction_id -> lifted_ir_id -> ssa_value_id -> skeleton_slot_id
```

这条链应从 lifter、IR builder 和 def-use 分析中保留下来。模型可以辅助不确定 slot role 或排序，但不能成为 grounding 的唯一来源。

---

## 2. Module B-v3 的任务边界

### 2.1 输入

Module B-v3 输入是机器可检查的函数级 IR 包：

```text
FunctionIR {
  function_id
  basic_blocks
  cfg_edges
  lifted_instructions
  ssa_values
  def_use_graph
  instruction_provenance
  branch_metadata
  call_metadata
  memory_access_summary
  upstream_confidence
}
```

每个 basic block 至少包含：

```json
{
  "bb_id": "bb_10",
  "address_range": ["0x401020", "0x40103a"],
  "instruction_ids": ["ins_1", "ins_2", "ins_3"],
  "terminator": "cond_branch",
  "has_call": false,
  "has_return": false,
  "side_effect_summary": ["flags_write", "memory_read"]
}
```

每条 CFG edge 至少包含：

```json
{
  "src": "bb_10",
  "dst": "bb_11",
  "kind": "true_branch",
  "condition_ssa": "v_flag_zf_13",
  "provenance_instruction": "ins_3",
  "confidence": 0.98
}
```

### 2.2 输出

输出不是源码 AST，而是可验证 binary structured skeleton：

```text
VerifiedSkeleton {
  skeleton
  slot_grounding
  escape_edges
  fallback_regions
  validator_report
  audit_log
}
```

最小节点类型：

```text
FUNC
SEQ
IF
IF_ELSE
LOOP
SWITCH
RETURN
BREAK
CONTINUE
BLOCK_SLOT
COND_SLOT
CALL_SLOT
GOTO_SLOT
OPAQUE_EXPR
ASM_EFFECT
UNSTRUCTURED_REGION
```

---

## 3. 总体架构

```text
Input FunctionIR
  |
  v
[B0] Deterministic Normalization and Analysis
  |
  v
[B1] Candidate Generator
  |
  v
[B2] Candidate Ranker, optional
  |
  v
[B3] Validator
  |
  +-- invalid -> repair / fallback
  |
  v
[B4] Canonicalizer
  |
  v
Verified Canonical Structured Skeleton
```

### 3.1 模块分工

| 模块 | 负责内容                                                  | 是否 ML |
| ---- | --------------------------------------------------------- | ------- |
| B0   | CFG/SSA 规范化、dominator、postdominator、SCC、loop、SESE | 否      |
| B1   | 生成合法候选和 fallback candidate                         | 否      |
| B2   | 对合法候选排序                                            | 可选 ML |
| B3   | CFG/SSA/provenance 验证                                   | 否      |
| B4   | canonicalization 和稳定输出                               | 否      |

关键工程约束：

```text
模型不能创建 CFG 边。
模型不能删除 CFG 边。
模型不能创建 SSA value。
模型不能绕过 validator。
模型不能阻止 fallback。
```

---

## 4. B0：确定性规范化与分析

### 4.1 CFG normalization

先将输入 CFG 规整成更适合结构化的形式。

必做：

1. 标准化 edge kind：

```text
fallthrough
true_branch
false_branch
backedge
call_return
tailcall
indirect_jump
exception_edge
unknown_edge
```

2. 标准化条件分支：

```text
CondBranch {
  block
  condition_ssa
  true_edge
  false_edge
  branch_instruction
  polarity
}
```

3. 标记异常/不支持控制流：

```text
indirect_jump unresolved
exception edge
setjmp/longjmp-like transfer
inline asm barrier
unknown noreturn
bad function boundary
```

4. 保留 provenance：

```text
bb -> instruction span
instruction -> lifted IR
lifted IR -> SSA def/use
branch edge -> condition SSA / branch instruction
call -> callee / args / return / clobber
```

### 4.2 静态分析结果

必须计算：

```text
dominators
postdominators
dominance_depth
postdominance_depth
SCCs
natural_loops
irreducible_components
SESE_regions
loop_nesting_forest
reachability
reverse_reachability
topological_order for DAG part
```

输出：

```text
AnalysisContext {
  normalized_cfg
  ssa
  provenance
  dom_tree
  postdom_tree
  sccs
  loops
  sese_regions
  unsupported_markers
}
```

---

## 5. B1：候选生成器

### 5.1 目标

Candidate generator 不生成唯一答案，而是为每个 region 生成 top-K 合法候选。

候选必须满足：

```text
grammar-valid
block-covered or fallback-covered
edge-preserving or escape-edge-preserving
slot schema fixed
provenance preserved
flattenable to CFG
```

### 5.2 候选类型

最小候选集合：

```text
SEQ_CANDIDATE
IF_THEN_CANDIDATE
IF_ELSE_CANDIDATE
GUARD_CANDIDATE
WHILE_CANDIDATE
DO_WHILE_CANDIDATE
LOOP_WITH_BREAK_CANDIDATE
LOOP_WITH_CONTINUE_CANDIDATE
SWITCH_CANDIDATE
RETURN_CANDIDATE
GOTO_PRESERVING_CANDIDATE
UNSTRUCTURED_REGION_CANDIDATE
```

### 5.3 候选 JSON schema

```json
{
  "candidate_id": "cand_001",
  "region_id": "region_10",
  "type": "IF_ELSE",
  "cover_bbs": ["bb_10", "bb_11", "bb_12"],
  "entry": "bb_10",
  "exits": ["bb_13"],
  "structured_edges": [
    ["bb_10", "bb_11"],
    ["bb_10", "bb_12"],
    ["bb_11", "bb_13"],
    ["bb_12", "bb_13"]
  ],
  "escape_edges": [],
  "slots": [
    {
      "slot_id": "slot_cond_001",
      "slot_type": "COND_SLOT",
      "allowed_grounding": ["v_cmp_12", "ins_33"],
      "default_grounding": "v_cmp_12"
    },
    {
      "slot_id": "slot_then_001",
      "slot_type": "BLOCK_SLOT",
      "cover_bbs": ["bb_11"]
    },
    {
      "slot_id": "slot_else_001",
      "slot_type": "BLOCK_SLOT",
      "cover_bbs": ["bb_12"]
    }
  ],
  "cost": {
    "goto_count": 0,
    "fallback_blocks": 0,
    "nesting_depth": 1,
    "condition_inversion": 0,
    "duplicated_blocks": 0
  },
  "validity_precheck": {
    "grammar_valid": true,
    "cfg_edges_explained": true,
    "ssa_grounding_available": true
  }
}
```

### 5.4 生成规则

#### 5.4.1 SEQ

适用条件：

```text
region 内子 region 可按 CFG 顺序线性连接。
每个子 region 的出口唯一进入下一个子 region。
无跨层跳转，或跨层跳转显式转为 GOTO_SLOT。
```

#### 5.4.2 IF / IF_ELSE

适用条件：

```text
存在条件分支 header h。
h 有 true_edge 和 false_edge。
then/else 区域满足单入口。
存在 join，或一个分支为 return/break/continue/goto。
condition_ssa 和 branch instruction 可定位。
```

候选变体：

```text
IF_THEN
IF_ELSE
IF with inverted condition
GUARD_CANDIDATE
```

#### 5.4.3 LOOP

适用条件：

```text
存在 natural loop 或 SCC。
有 header、body、latch、exit evidence。
loop body 内部 edge 可解释。
break/continue target 可由 edge 归类。
```

候选变体：

```text
WHILE
DO_WHILE
FOR_LIKE, 仅作为 presentation hint
LOOP_WITH_BREAK
LOOP_WITH_CONTINUE
UNSTRUCTURED_REGION for irreducible loop
```

#### 5.4.4 SWITCH

MVP 可先弱支持。

适用条件：

```text
存在 indirect jump 或 branch chain。
有 switch key SSA。
有 target set proof。
case target 与 CFG edge 一致。
default edge 可解释。
```

若 target set 不完整：

```text
输出 INDIRECT_JUMP_SLOT 或 UNSTRUCTURED_REGION。
```

#### 5.4.5 Fallback

每个 region 必须生成 fallback：

```json
{
  "type": "UNSTRUCTURED_REGION",
  "cover_bbs": ["bb_10", "bb_11", "bb_12"],
  "labels": ["L10", "L11", "L12"],
  "gotos": [
    ["L10", "L11"],
    ["L10", "L12"],
    ["L11", "L12"]
  ],
  "reason": "no_valid_structured_candidate"
}
```

这保证系统永不因模型失败而输出非法 skeleton。

### 5.5 候选生成伪代码

```text
Algorithm CandidateGeneration(region R, AnalysisContext A)
Input:
  R: CFG region
  A: normalized CFG, SSA, dominance, loops, provenance
Output:
  C: legal candidate list

1. C = []
2. if is_linear_sequence(R, A):
       C.add(make_seq_candidate(R))
3. for each branch header h in R:
       if is_valid_if_region(h, R, A):
           C.add(make_if_candidate(h, R, polarity=normal))
           C.add(make_if_candidate(h, R, polarity=inverted))
           if is_guard_style(h, R, A):
               C.add(make_guard_candidate(h, R))
4. for each loop L in loops_inside(R):
       if is_natural_loop(L):
           C.add(make_while_candidate(L))
           C.add(make_do_while_candidate(L))
           C.add(make_loop_with_break_continue_candidate(L))
       else:
           C.add(make_unstructured_region_candidate(L, reason=irreducible))
5. for each switch evidence S in R:
       if has_target_set_proof(S):
           C.add(make_switch_candidate(S))
       else:
           C.add(make_indirect_jump_fallback(S))
6. C.add(make_unstructured_region_candidate(R, reason=fallback))
7. filter candidates by grammar and CFG precheck
8. deduplicate equivalent candidates by canonical key
9. return top-K candidates by deterministic cost
```

---

## 6. B2：小模型排序器

### 6.1 第一版是否必须有模型

不必须。

实施顺序应是：

```text
MVP-0: deterministic candidate generator + validator
MVP-1: add candidate ranker
MVP-2: add slot pointer, if needed
```

如果 MVP-0 不稳定，不应训练模型。

### 6.2 模型输入

#### Block features

```text
opcode_family_histogram
block_instruction_count
terminator_kind
has_call
has_return
has_indirect_jump
has_memory_write
has_volatile_or_barrier
in_degree
out_degree
dominance_depth
postdominance_depth
loop_depth
ssa_def_count
ssa_use_count
side_effect_summary
```

#### Edge features

```text
edge_kind
is_fallthrough
is_true_branch
is_false_branch
is_backedge
is_exception
is_indirect
condition_ssa_available
branch_instruction_available
```

#### Candidate features

```text
candidate_type
node_type_sequence
covered_block_count
structured_edge_count
escape_edge_count
goto_count
fallback_block_count
nesting_depth
condition_inversion_count
duplicated_block_count
loop_kind
switch_case_count
repair_needed_count
validator_precheck_flags
```

### 6.3 模型结构

第一版建议非常小：

```text
CFG Encoder:
  2-layer GraphSAGE / R-GCN

Candidate Encoder:
  MLP over candidate features
  or tiny transformer over skeleton node tokens

Scorer:
  score = MLP([region_embedding, candidate_embedding, cost_features])
```

可选 slot pointer：

```text
slot_repr = MLP([candidate_repr, slot_type_embedding, covered_block_pool])
target_repr = CFG/SSA object embedding
P(target | slot) = masked_softmax(slot_repr dot target_repr)
```

### 6.4 第一版 heads

只保留：

```text
candidate_score_head
```

第二阶段可加：

```text
slot_pointer_head
```

不要在第一版加入：

```text
parent_head
role_head
production_head
region_head
consistency_head
soft_cfg_head
OT_alignment_head
Hungarian_set_prediction_head
```

---

## 7. 训练目标

### 7.1 第一版主 loss：L_rank

候选集合：

```text
C = CandidateGeneration(G_b, SSA)
C = C union {fallback candidates}
```

模型：

```text
p(c_i | I_b) = softmax(score_theta(c_i, I_b))
```

训练标签：

```text
c_best = best legal candidate under proxy teacher
```

主 loss：

```text
L_rank = - w_teacher * log p(c_best | I_b)
```

如果有多个接近等价的候选，则使用 soft label：

```text
q_i = softmax(quality(c_i) / tau)
L_rank = CE(q, p)
```

其中：

```text
quality(c) =
  alpha_type   * type_match(c, teacher)
+ alpha_region * region_IoU(c, teacher)
+ alpha_edge   * edge_explained(c, G_b)
+ alpha_slot   * slot_template_match(c, teacher)
- alpha_fb     * fallback_penalty(c)
- alpha_goto   * goto_penalty(c)
```

### 7.2 Teacher-to-candidate 标签构建

训练期可用源码 AST / debug / alignment，但只作为 preference signal。

标签构建流程：

```text
Algorithm BuildCandidateLabels(I_b, T_proxy, C)
1. for each candidate c in C:
       if not precheck_valid(c):
           quality(c) = -inf
       else:
           quality(c) = weighted_match(c, T_proxy, G_b)
2. if max quality(c) < threshold:
       c_best = fallback_candidate
       w_teacher = low_or_medium
3. else:
       c_best = argmax quality(c)
       w_teacher = teacher_confidence * candidate_match_confidence
4. return c_best or soft distribution q
```

过滤规则：

```text
teacher_confidence < 0.7 -> 不做 hard label
alignment missing -> soft label or fallback positive
optimized mismatch severe -> lower weight
candidate top-K no good match -> fix candidate generator first
```

### 7.3 第二阶段 grounding loss

不和结构训练混在一起。

先使用确定性规则绑定：

```text
COND_SLOT -> branch instruction / flags SSA / true-false edge
BLOCK_SLOT -> instruction span
CALL_SLOT -> call instruction / callee / args / return / clobber
OPAQUE_REGION -> entry/exit edges + internal BBs
```

如果需要训练 slot pointer：

```text
L_ground =
  CE(condition_ssa)
+ CE(condition_polarity)
+ BCE(instruction_span_coverage)
```

如果一个 slot 有多个等价 grounding：

```text
L = -log sum_{t in positive_targets} P(t | slot)
```

不要第一版引入：

```text
memory SSA KL
OT grounding
parent-slot joint loss
global consistency loss
```

---

## 8. Validator

validator 是系统最终裁判，不是离线评估工具。

### 8.1 必检项

```text
schema_validity
single_root
tree_or_allowed_escape_dag
slot_id_unique
block_coverage
instruction_coverage
cfg_edge_preservation
escape_edge_explicitness
flattened_cfg_equivalence
condition_grounding
condition_polarity
ssa_value_existence
def_use_legality
side_effect_ordering
fallback_region_boundary
canonical_serialization_parse
```

### 8.2 CFG equivalence 检查

流程：

```text
Algorithm ValidateControlEquivalence(S, G)
1. G_s = LowerSkeletonToCFG(S)
2. normalize labels in G_s and G
3. compare reachable block set
4. compare edge set by block id and edge kind
5. for conditional edges:
       compare condition_ssa and polarity, allowing explicit inversion record
6. for fallback regions:
       check all internal edges are preserved inside region
7. for escape edges:
       check each escape corresponds to original CFG edge
8. return pass or detailed diff
```

失败报告必须包含：

```json
{
  "error_type": "missing_edge",
  "region": "region_10",
  "candidate": "cand_003",
  "expected": ["bb_12", "bb_15"],
  "actual": null,
  "suggested_action": "fallback_region"
}
```

### 8.3 Validator 后处理策略

```text
if top1 candidate passes:
    accept
else:
    try top2...topK
if no structured candidate passes:
    emit fallback candidate
if fallback fails:
    emit labeled CFG pseudocode
```

fallback 后硬指标仍应满足：

```text
BB coverage = 100%
CFG edge preservation = 100%
instruction provenance complete
```

---

## 9. Canonicalizer

canonicalizer 负责稳定输出，减少等价结构的随机性。

规则：

```text
1. flatten nested SEQ
2. remove empty ELSE
3. prefer fallthrough as THEN or BODY according to fixed rule
4. normalize condition inversion
5. sort switch cases by case value
6. normalize loop header and latch naming
7. normalize break / continue target labels
8. stable label naming by block address or stable block id
9. collapse single-child structured region
10. preserve explicit GOTO only when needed for CFG equivalence
```

canonicalizer 不能改变 CFG 语义。canonicalize 后必须重新跑 validator。

---

## 10. 推理流程

```text
Algorithm ModuleBv3Infer(FunctionIR f)
Input:
  f: binary function IR
Output:
  verified canonical skeleton

1. A = NormalizeAndAnalyze(f)
2. R = BuildRegionForest(A)
3. for each region r bottom-up:
       C_r = CandidateGeneration(r, A)
       C_r = PrecheckAndDeduplicate(C_r)
       if model available:
           score each c in C_r
           order C_r by model score + deterministic cost
       else:
           order C_r by deterministic cost
       selected_r = first candidate passing validator local checks
       if none:
           selected_r = fallback candidate
4. compose region candidates into whole-function skeleton S
5. run whole-function validator
6. if invalid:
       repair bounded errors
7. if still invalid:
       fallback to labeled CFG skeleton
8. S_canon = Canonicalize(S)
9. run validator again
10. emit S_canon + grounding + audit log
```

---

## 11. 训练流程

### Stage 0：候选生成器验证，不训练模型

目标：确认 ranker 有意义。

指标：

```text
top-16 oracle skeleton accuracy >= 90% initially, target 95%
top-32 oracle grounding-compatible accuracy >= 85% initially, target 90%
average candidate count <= 32
fallback candidate coverage = 100%
invalid candidate entering training = 0
```

如果 top-K 里没有好候选，不要训练模型，先修 candidate generator。

### Stage 1：只训练 candidate ranker

输入：

```text
(FunctionIR, candidates, c_best or soft quality distribution)
```

Loss：

```text
L = L_rank
```

训练数据顺序：

```text
simple sequence
if / if_else
nested if
while / do_while
break / continue
switch
real small functions
```

验收：

```text
rank loss 下降
top-1 candidate accuracy 高于 deterministic cost baseline
top-3 包含最佳候选比例提升
validator pass rate 不下降
fallback overuse 不上升
```

### Stage 2：slot grounding 单独训练

先用规则 provenance 绑定，再只对高置信样本训练轻量 pointer。

Loss：

```text
L = L_ground_basic
```

验收：

```text
instruction grounding coverage >= 95%
COND_SLOT polarity error 低于 BB-only baseline
SSA def-use validator fatal error = 0 after fallback
```

### Stage 3：n-best reranking

只有 Stage 1/2 稳定后再做。

输入不是原始图，而是 validator 通过的 n-best audit features：

```text
model_score
fallback_count
goto_count
repair_count
nesting_depth
structured_coverage
grounding_confidence
condition_inversion_count
```

Loss：

```text
L = L_nbest_rank
```

目标：

```text
降低 fallback
减少 repair
提高可读性
保持 CFG equivalence
```

---

## 12. 数据构建

### 12.1 Synthetic-controlled 数据

自动生成 C subset：

```text
sequence
if
if_else
nested_if
while
do_while
for_like
break
continue
early_return
switch_simple
```

编译矩阵：

```text
Compiler: gcc, clang
Opt: -O0, -O1, -O2
Arch: x86-64 first
Debug info: training label only, inference forbidden
```

用途：

```text
debug candidate generator
debug validator
train clean ranker
measure oracle recall
```

### 12.2 Real small functions

从 Coreutils、BusyBox、SQLite 等项目抽取小函数。

过滤：

```text
3 <= BB count <= 50
exclude unresolved indirect jump initially
exclude exception-heavy functions initially
exclude inline asm initially
exclude suspected bad function boundary
```

用途：

```text
验证不是 toy-only
统计 failure buckets
评估 fallback rate
```

### 12.3 Stress set

包含：

```text
large function
deep nesting
multi-exit loop
switch-heavy
irreducible CFG
unresolved indirect jump
exception edge
tailcall / noreturn
bad CFG perturbation
```

Stress set 不作为第一版 pass gate，只用于边界分析。

---

## 13. Baselines

### B0：CFG-to-goto baseline

直接输出 labeled block + goto。

预期：

```text
CFG equivalence high
structuredness poor
```

作用：

```text
验证 lowering / validator 闭环。
证明结构化收益不是来自简单保边。
```

### B1：classic structurer baseline

基于 dominator、natural loop、region collapse。

作用：

```text
Module B-v3 必须至少不明显弱于它。
```

### B2：existing decompiler baseline

可选：

```text
Ghidra
angr / AIL
RetDec
Hex-Rays, if available
```

只比较 skeleton 层，不比较完整源码文本。

---

## 14. 验收指标

### 14.1 第一优先级：正确性

```text
BB coverage = 100%
CFG edge preservation = 100%, or all missing edges explicit fallback
accepted cases CFG equivalence = 100%
False verified rate = 0
validator fatal error = 0 after fallback
skeleton parse success >= 98%
timeout rate < 1%
```

### 14.2 第二优先级：结构化程度

```text
Verified Skeleton Rate
Fallback Region Rate
GOTO count per function
Structured block coverage
Average nesting depth
Repair count per function
```

### 14.3 第三优先级：模型效果

```text
candidate top-1 accuracy
candidate top-3 oracle hit
ranker improvement over deterministic cost
slot grounding accuracy
condition polarity error
```

注意：第一版不要用下面指标作为 hard gate：

```text
source AST exact match
token-level exact match
LLM judge score
完整反编译源码 compile rate
复杂 learned loss 曲线
```

---

## 15. 一周 MVP 排期

| 时间  | 任务                                          | 产出                                               | 验收                               |
| ----- | --------------------------------------------- | -------------------------------------------------- | ---------------------------------- |
| Day 1 | 固化 IR 输入、skeleton JSON、validator schema | `FunctionIR`、`Candidate`、`Skeleton` schema | 能 parse 100 个函数 IR             |
| Day 2 | deterministic structurer + fallback           | SEQ/IF/LOOP/UNSTRUCTURED 基线                      | 无模型也能保 CFG 输出              |
| Day 3 | candidate generator                           | if/loop/switch/seq/opaque/goto candidates          | 每个 unresolved region 有 fallback |
| Day 4 | teacher-to-candidate label builder            | high-confidence labels                             | 低置信样本不 hard train            |
| Day 5 | 训练 ranker                                   | `L_rank` 模型                                    | loss 下降，top-k 可解释            |
| Day 6 | validator + repair + fallback                 | local/global validator                             | validator error 自动降级           |
| Day 7 | 跑 100-1000 函数                              | failure buckets + metrics                          | 无 crash，审计日志完整             |

---

## 16. Stop / Go 决策

| Gate                | Go 条件                                  | Stop 条件                    | 下一步                 |
| ------------------- | ---------------------------------------- | ---------------------------- | ---------------------- |
| G0 Verifier         | 正负例稳定判断，错误可定位               | verifier 有假阳性            | 修 verifier            |
| G1 Candidate Recall | top-16 oracle >= 90%                     | 好候选召回低                 | 修 candidate generator |
| G2 Synthetic        | verified >= 90%, accepted edge eq = 100% | 简单结构都不稳               | 修 structurer          |
| G3 Real Small       | verified >= 60%, 失败可归因              | 真实函数失败不可归因         | 改输入规范或诊断       |
| G4 Ranker           | 超过 deterministic cost baseline         | ranker 不提升或增加 fallback | 暂停 ML，仅保规则版    |
| G5 Grounding        | instruction grounding >= 95%             | grounding 破坏 SSA 检查      | 回到规则 provenance    |
| G6 Add Complexity   | 瓶颈明确且简单方案到顶                   | 只是感觉需要模型             | 不加复杂 loss          |

---

## 17. 实施清单

### 必做

```text
[ ] FunctionIR schema
[ ] block / edge / instruction / SSA stable ids
[ ] instruction provenance chain
[ ] CFG normalization
[ ] dominator / postdominator analysis
[ ] SCC / natural loop analysis
[ ] SESE or interval region extraction
[ ] skeleton JSON schema
[ ] candidate JSON schema
[ ] deterministic candidate generator
[ ] fallback candidate for every region
[ ] skeleton lowering to CFG
[ ] CFG edge equivalence validator
[ ] block / instruction coverage validator
[ ] SSA / slot grounding validator
[ ] canonicalizer
[ ] audit log schema
[ ] failure bucket reporter
```

### MVP 后再做

```text
[ ] candidate ranker
[ ] slot pointer head
[ ] n-best reranker
[ ] switch advanced support
[ ] irreducible CFG handling beyond fallback
[ ] Module C/D feedback local re-ranking
```

### 不建议第一版做

```text
[ ] end-to-end skeleton decoder
[ ] Hungarian matching loss
[ ] UOT loss
[ ] parent / role / production heads
[ ] soft CFG loss
[ ] consistency loss
[ ] RL verifier reward
[ ] LLM-generated skeleton
[ ] full source AST recovery claim
```

---

## 18. 最终可宣称内容

如果按本方案完成 MVP，可以保守宣称：

> Module B-v3 在 binary-only CFG/SSA 输入下，通过确定性候选生成、候选排序、CFG/SSA validator 和 fallback，生成可验证的 canonical structured skeleton；在支持范围内保证 accepted outputs 的 CFG 等价，超出范围时显式 fallback 而不是静默输出错误结构。

不能宣称：

```text
恢复源码 AST
解决反编译
恢复唯一真实控制结构
模型学会 CFG 合法性
对混淆/异常/所有优化等级都鲁棒
```

---

## 19. 最短落地路线

最短路线不是训练模型，而是先跑通闭环：

```text
1. FunctionIR
2. Candidate schema
3. Fallback skeleton
4. Lower skeleton to CFG
5. Validator
6. Deterministic structurer
7. Candidate generator
8. Failure buckets
9. Ranker
10. Slot grounding pointer
```

只要前 8 步稳定，系统已经具备工程价值；第 9 和第 10 步只是提升结构偏好和可读性，不承担正确性兜底。
