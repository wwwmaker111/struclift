/**
 * StrucLift：数据构建 → 训练 → 测试（详细版）
 * 依据 Research Proposal §6–§10 与仓库训练脚本惯例
 *
 * Usage: node build_train_exp_flowchart_pptx.cjs [output.pptx]
 */
'use strict';

const path = require('path');
const pptxgen = require('pptxgenjs');

const OUT =
  process.argv[2] ||
  path.join('E:', 'user', 'Downloads', 'struclift_train_to_experiment_flowchart.pptx');

const FONT = 'Microsoft YaHei';
const TITLE_C = '141413';
const SUB_C = '3D3D3A';
const ACC_DATA = '5F4FB8';
const ACC_TRAIN = '0F6E56';
const ACC_TEST = '6D4F11';

function sectionBox(slide, pres, opts) {
  const {
    x, y, w, h,
    fill, line,
    heading,
    lines,
    headingColor,
    fontBody = 9.8,
    lineSpacing = 1.32,
    charSp = 0.25,
  } = opts;

  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h,
    fill: { color: fill },
    line: { color: line, width: 1.15 },
    rectRadius: 0.06,
  });

  const parts = [{
    text: heading + '\n',
    options: {
      fontSize: 11.4,
      bold: true,
      color: headingColor || TITLE_C,
      fontFace: FONT,
      breakLine: true,
    },
  }];
  lines.forEach((line, i) => {
    parts.push({
      text: i < lines.length - 1 ? line + '\n' : line,
      options: {
        fontSize: fontBody,
        color: SUB_C,
        fontFace: FONT,
        breakLine: i < lines.length - 1,
      },
    });
  });

  slide.addText(parts, {
    x: x + 0.11,
    y: y + 0.12,
    w: w - 0.22,
    h: h - 0.2,
    valign: 'top',
    lineSpacingMultiple: lineSpacing,
    charSpacing: charSp,
    margin: 5,
  });
}

function slideHeader(slide, main, sub) {
  slide.addText(main, {
    x: 0.45,
    y: 0.28,
    w: 9.1,
    h: 0.52,
    fontSize: 22,
    bold: true,
    fontFace: FONT,
    color: TITLE_C,
    align: 'center',
    charSpacing: 0.35,
  });
  if (sub) {
    slide.addText(sub, {
      x: 0.45,
      y: 0.78,
      w: 9.1,
      h: 0.36,
      fontSize: 11,
      fontFace: FONT,
      color: SUB_C,
      align: 'center',
    });
  }
}

function main() {
  const pres = new pptxgen();
  /** 略高于 16:9，容纳多栏流程文字（inch） */
  pres.defineLayout({ name: 'FLOW_WIDE_TALL', width: 10, height: 11 });
  pres.layout = 'FLOW_WIDE_TALL';
  pres.title = 'StrucLift 数据—训练—测试流程';
  pres.author = 'StrucLift';

  // ─── Slide 1: 数据构建 ──────────────────────────────────────
  const s1 = pres.addSlide();
  s1.background = { color: 'FAFAFA' };
  slideHeader(s1, '一、数据构建流水线（仓库实现）', 'build_binskel_dataset_md.py · build_*_binskel.sh · binskel_jsonl 字段契约');

  sectionBox(s1, pres, {
    x: 0.42,
    y: 1.35,
    w: 4.55,
    h: 4.15,
    fill: 'EDEAFB',
    line: '5349B7',
    heading: '① 主入口脚本与编译矩阵',
    headingColor: ACC_DATA,
    fontBody: 9.35,
    lines: [
      '主构建器：**`build_binskel_dataset_md.py`**（头文件说明：DWARF 行表 + **LLVM IR `!dbg`** 双信号交叉验证 → 高置信对齐）。',
      '项目级封装：仓库 **`scripts/build_<工程>_binskel.sh`**（WSL/Linux bash），内部调用上述 Python；**`BUILD_BINSKEL_COMMANDS.md`** 给出手写参数模板。',
      '每优化档输入三件套：**ELF** + **同源树 `--src`** + **同档合并 `--llvm-ir …ll`** + `--opt O0…Os`（高优化必须带对应 IR，勿 `--skip-alignment`）。',
      '规模化惯例：**九项目 × 五档（O0–O3+Os）→ 约 45 份 `binskel_<proj>_o*.jsonl`**（见 **`docs/build_binskel_9x5_full.md`**）；**coreutils / ffmpeg / openssh** 等为 **multibin** 多 ELF 合并。',
    ],
  });

  sectionBox(s1, pres, {
    x: 5.08,
    y: 1.35,
    w: 4.48,
    h: 4.15,
    fill: 'EDEAFB',
    line: '5349B7',
    heading: '② 二进制 CFG 张量（与 encode 对齐）',
    headingColor: ACC_DATA,
    fontBody: 9.35,
    lines: [
      '**Capstone** 反汇编 → 按 BB 切段：**bin_opcodes**，**bin_operand_types / bin_operand_values**（长度 n_instr×3），**bin_block_lengths**。',
      '图结构：**bin_edge_index** = [src[], dst[]]；**bin_edge_type** 与边一一对应（cond/uncond/back/call/ret）。',
      '**bin_struct_features**：每 BB 一行、**固定 12 维**结构特征（与 `binskel_jsonl._validate_sample` 一致）。',
      '监督：**pattern_labels**，长度 = **n_bb**；元数据：**binary / func_name / opt_level / source_file** 等。',
    ],
  });

  sectionBox(s1, pres, {
    x: 0.42,
    y: 5.58,
    w: 4.55,
    h: 4.68,
    fill: 'F5F3FF',
    line: '5349B7',
    heading: '③ 对齐矩阵与源码侧（MD 流水线）',
    headingColor: ACC_DATA,
    fontBody: 9.35,
    lines: [
      '输出对齐：**block_src_lines**；**alignment_matrix**，形状 **[n_bb][n_stmt]**；可选 **alignment_confidence**（与矩阵同形）、**block_src_confidence**。',
      '源码 **tree-sitter 语句档**：**src_token_ids** 与 **src_stmt_lengths**；**src_ast_types**；**src_edge_index / src_edge_type**；**src_compound_id**（与语句列对齐）。',
      '质量标签：**src_encoding_tier**（语句档 stmt / 行级回退 line_fallback）；默认 **丢弃 line_fallback**（需旧行为时 `--keep-line-fallback`）；**bb_alignment_coverage** 可统计。',
      '对照：旧版 **`build_binskel_dataset.py`** 为简化路径；生产以 **`_md.py`** 为准。',
    ],
  });

  sectionBox(s1, pres, {
    x: 5.08,
    y: 5.58,
    w: 4.48,
    h: 4.68,
    fill: 'F5F3FF',
    line: '5349B7',
    heading: '④ 校验、合并与下游样本',
    headingColor: ACC_DATA,
    fontBody: 9.35,
    lines: [
      '字段契约：读入 **`struclift/data/binskel_jsonl.py`**（**_validate_sample**：无源码/无对齐/SFT 长度等门槛）。读取可 **mmap**（环境变量 **`STRUCLIFT_NO_JSONL_MMAP=1` 关闭**）。',
      '独立校验：**`scripts/validate_alignment.py`**（BB 覆盖率、矩阵与图一致性等）。',
      '合并：多档 jsonl 可 **cat** 或按 `(func_name,opt)` 去重（见 BUILD_BINSKEL 文档 §5）；**跨机路径**：**STRIP_PATH_PREFIX** 传给脚本统一前缀。',
      'Stage3：**`scripts/augment_binskel_sft.py`** 写入 **sft_input_ids / sft_labels / slot_types**（及 RL 可选字段）。Stage2：**`train_stage2_binskel.py --curriculum --jsonl-o0…`**。',
      'Stage1 数据集：**`cross_opt_pairs`** 启用同函数跨优化配对（见 BinSkelJsonlDataset 注释）；**HardNegativeBank** 在 **Stage2 训练器**内维护，非 jsonl 静态字段。',
    ],
  });

  // ─── Slide 2: 训练 ───────────────────────────────────────────
  const s2 = pres.addSlide();
  s2.background = { color: 'FAFAFA' };
  slideHeader(s2, '二、四阶段训练流水线', 'checkpoint 逐级初始化 · 损失与冻结策略（§9）');

  sectionBox(s2, pres, {
    x: 0.42,
    y: 1.28,
    w: 9.14,
    h: 1.92,
    fill: 'E8F7F2',
    line: '0F6E56',
    heading: 'Stage 1 · Module A 预训练（表示学习与 pattern）',
    headingColor: ACC_TRAIN,
    lines: [
      '损失（实现见 stage1_pretrain）：L_pattern + λ_recon·L_neighbor（InfoNCE 邻域）+ 可选 λ_graph·L_graph_bin_src + λ_region·L_region_InfoNCE。',
      '训练对象：BinaryCFGEncoder + SourceCFGEncoder。产出：best_stage1.pt / latest_stage1.pt（完整 ckpt 含优化器）。',
    ],
  });

  sectionBox(s2, pres, {
    x: 0.42,
    y: 3.32,
    w: 9.14,
    h: 2.08,
    fill: 'E8F7F2',
    line: '0F6E56',
    heading: 'Stage 2 · Module B 对齐（冻结 Module A）',
    headingColor: ACC_TRAIN,
    lines: [
      '损失：L_SCOT + λ_edge·L_edge + λ_region·L_region + λ_ca·L_ca_align + λ_src·L_src_count + 可选节点硬负。',
      'Curriculum：先易后难 O0→O1→O2→O3→Os（s2_curriculum_epochs_per_stage 控制切档）；验证可按 curriculum stage 换 val 集。',
      '产出：best_stage2.pt；可分 best_stage2_curriculum_stage{k}.pt。',
    ],
  });

  sectionBox(s2, pres, {
    x: 0.42,
    y: 5.48,
    w: 4.48,
    h: 1.92,
    fill: 'FAEFE9',
    line: '993C1D',
    heading: 'Stage 3 · Module C SFT',
    headingColor: '993C1D',
    lines: [
      '冻结 A+B；Adapter + LoRA；L_fill（加权 slot）+ λ_compile·L_compile；Teacher forcing + Scheduled sampling。',
      '产出：best_stage3.pt（完整 StrucLift，用于推理与 Stage4 初始化）。',
    ],
  });

  sectionBox(s2, pres, {
    x: 5.08,
    y: 5.48,
    w: 4.48,
    h: 1.92,
    fill: 'FAEFE9',
    line: '993C1D',
    heading: 'Stage 4 · Module D GRPO RL',
    headingColor: '993C1D',
    lines: [
      '每样本采样 G 条候选；奖励 α·R_compile + β·R_struct + γ·R_semantic；KL(π‖π_ref)，π_ref 常为 CPU 冻结副本。',
      '更新：LoRA+Adapter 为主；可选 finetune_module_b_cross_attn @ lr×0.1。',
    ],
  });

  // ─── Slide 3: 测试 ───────────────────────────────────────────
  const s3 = pres.addSlide();
  s3.background = { color: 'FAFAFA' };
  slideHeader(s3, '三、测试与实验评测', '离线推理管线 · 基准 · 指标 · 消融（§10）');

  sectionBox(s3, pres, {
    x: 0.42,
    y: 1.28,
    w: 9.14,
    h: 1.75,
    fill: 'EFF6E4',
    line: '5F9117',
    heading: '① 离线推理 / 评测管线',
    headingColor: ACC_TEST,
    lines: [
      '加载 checkpoint → 构造二进制 CFG batch → encode(run_alignment=true) 得 graph_kv / skeleton → Module C 自回归解码生成 C/C++。',
      '可与官方反编译器（Ghidra）或端到端 baseline 在同输入上对比：编译 gcc、提取生成 CFG、跑测试用例。',
    ],
  });

  sectionBox(s3, pres, {
    x: 0.42,
    y: 3.14,
    w: 4.48,
    h: 2.65,
    fill: 'F3F9EC',
    line: '5F9117',
    heading: '② 数据集与泛化套件',
    headingColor: ACC_TEST,
    lines: [
      '训练规模：提案 BinSkel-Train（多项目 × 多编译配置）。',
      '公开对比：Decompile-Bench-Eval；HumanEval-Decompile。',
      '自建泛化：Cross-Opt（训 O0/O1 测 O2/O3）；Cross-Compiler（训 GCC 测 Clang）；Obfuscation（OLLVM 等）。',
    ],
  });

  sectionBox(s3, pres, {
    x: 5.08,
    y: 3.14,
    w: 4.48,
    h: 2.65,
    fill: 'F3F9EC',
    line: '5F9117',
    heading: '③ 指标体系',
    headingColor: ACC_TEST,
    lines: [
      '结构（核心）：CSA；TED；CFG Edge F1；Node Alignment F1。',
      '功能对齐 SOTA：Re-compilability；Re-executability；Edit Similarity；BLEU / CodeBLEU。',
      '资源：推理延迟（ms/函数）；显存（GB）。',
    ],
  });

  sectionBox(s3, pres, {
    x: 0.42,
    y: 5.88,
    w: 9.14,
    h: 2.42,
    fill: 'F7FBE8',
    line: '8A6F0A',
    heading: '④ 对比方法 · 消融 · 图表产出',
    headingColor: ACC_TEST,
    fontBody: 9.4,
    lines: [
      '对比：Ghidra；LLM4Decompile；Nova；SK2Decompile；SALT；CodeInverter；D-LiFT（§10.3）。',
      '消融（§10.4）：去 pattern 头 / SCOT / Region / Topology 正则 / Cross-Attn 连续注入 / Struct RL / Curriculum 等。',
      '论文级图表：对齐矩阵 M 热力图；O0→O3 指标曲线；消融表；端到端 Case Study（结构+代码对照）。',
    ],
  });

  return pres.writeFile({ fileName: OUT, compression: true });
}

main()
  .then(() => console.log('Wrote', OUT))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
