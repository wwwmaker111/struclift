/**
 * 两页「四阶段训练流程」：Slide1 = Stage1+2，Slide2 = Stage3+4。
 * 不含底部 HardNegativeBank / s1_cross_opt_pairs 说明行；行距、字距放宽便于阅读。
 *
 * Usage: node build_four_stage_training_pptx.cjs [output.pptx]
 */
'use strict';

const path = require('path');
const pptxgen = require('pptxgenjs');

const OUT =
  process.argv[2] ||
  path.join('E:', 'user', 'Downloads', 'struclift_four_stage_AB_CD.pptx');

const FONT = 'Microsoft YaHei';
const FILL = 'F1EFE8';
const STROKE = '5F5E5A';
const TEXT_MAIN = '444441';
const TEXT_MUTED = '5F5E5A';
const TITLE_COLOR = '141413';

/** 行距倍数、字间距（pt），略放大以减轻紧凑感 */
const LINE_MULT = 1.48;
const CHAR_SP = 1.05;
const BODY_PT = 13.5;
const HEAD_PT = 15.5;
const FOOT_PT = 12.5;

function stageBox(slide, pptx, { x, y, w, h }, lines) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    fill: { color: FILL },
    line: { color: STROKE, width: 1 },
    rectRadius: 0.12,
  });

  const parts = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const isHead = i === 0;
    parts.push({
      text: i < lines.length - 1 ? line + '\n' : line,
      options: {
        fontSize: isHead ? HEAD_PT : BODY_PT,
        bold: isHead,
        breakLine: i < lines.length - 1,
        color: TEXT_MAIN,
        fontFace: FONT,
      },
    });
  }

  slide.addText(parts, {
    x: x + 0.18,
    y: y + 0.22,
    w: w - 0.36,
    h: h - 0.4,
    align: 'center',
    valign: 'middle',
    lineSpacingMultiple: LINE_MULT,
    charSpacing: CHAR_SP,
    margin: 10,
  });
}

function hArrow(slide, pptx, x0, y0, x1) {
  const minX = Math.min(x0, x1);
  const w = Math.max(0.06, Math.abs(x1 - x0));
  const h = 0.02;
  slide.addShape(pptx.ShapeType.line, {
    x: minX,
    y: y0 - h / 2,
    w,
    h,
    line: { color: '73726C', width: 1.75, endArrowType: 'arrow' },
  });
}

function footer(slide, text, cx, y) {
  const w = 4.2;
  slide.addText(text, {
    x: cx - w / 2,
    y,
    w,
    h: 0.45,
    fontSize: FOOT_PT,
    fontFace: FONT,
    color: TEXT_MUTED,
    align: 'center',
    valign: 'top',
    lineSpacingMultiple: 1.35,
    charSpacing: 0.85,
  });
}

function main() {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  pres.title = 'StrucLift 四阶段训练流程';

  const boxW = 3.88;
  const boxH = 3.55;
  const boxY = 1.02;
  const box1X = 0.62;
  const box2X = 5.48;
  const midY = boxY + boxH / 2;

  const mkSlide = (title, leftLines, rightLines, leftFoot, rightFoot) => {
    const slide = pres.addSlide();
    slide.background = { color: 'FFFFFF' };

    slide.addText(title, {
      x: 0.4,
      y: 0.32,
      w: 9.2,
      h: 0.75,
      fontSize: 22,
      bold: true,
      fontFace: FONT,
      color: TITLE_COLOR,
      align: 'center',
      valign: 'middle',
      charSpacing: 0.6,
    });

    stageBox(slide, pres, { x: box1X, y: boxY, w: boxW, h: boxH }, leftLines);
    stageBox(slide, pres, { x: box2X, y: boxY, w: boxW, h: boxH }, rightLines);

    const xArrow0 = box1X + boxW + 0.06;
    const xArrow1 = box2X - 0.06;
    hArrow(slide, pres, xArrow0, midY, xArrow1);

    const footY = boxY + boxH + 0.18;
    footer(slide, leftFoot, box1X + boxW / 2, footY);
    footer(slide, rightFoot, box2X + boxW / 2, footY);
  };

  mkSlide(
    '四阶段训练流程',
    ['Stage 1', 'Module A 预训练', 'L_pattern', '+ graph对比'],
    ['Stage 2', 'Module B 对齐训练', '+SCOT+Edge+Region', 'Curriculum O0→O3'],
    '训练: A',
    '冻结: A · 训练: B'
  );

  mkSlide(
    '四阶段训练流程',
    ['Stage 3', 'Module C SFT', '冻结 A+B · 训练', 'LoRA + Adapter'],
    ['Stage 4', 'GRPO RL 优化', '结构一致性奖励', 'ref_model 冻结'],
    '冻结: A+B · 训练: C',
    '训练: C+B(×0.1)'
  );

  return pres.writeFile({ fileName: OUT, compression: true });
}

main()
  .then(() => console.log('Wrote', OUT))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
