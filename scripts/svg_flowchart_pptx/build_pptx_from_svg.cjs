/**
 * Converts struclift_accurate_flowchart.svg to PPTX with native shapes + text boxes.
 * Usage: node build_pptx_from_svg.cjs [input.svg] [output.pptx]
 */
'use strict';

const fs = require('fs');
const path = require('path');
const { DOMParser } = require('@xmldom/xmldom');
const pptxgen = require('pptxgenjs');

const SVG_IN = process.argv[2] || 'E:\\user\\Downloads\\struclift_accurate_flowchart.svg';
const PPTX_OUT =
  process.argv[3] || 'E:\\user\\Downloads\\struclift_accurate_flowchart_editable.pptx';

const VB_W = 680;
const VB_H = 1380;
const SLIDE_W = 7.5;
const SLIDE_H = (VB_H / VB_W) * SLIDE_W;
const sx = SLIDE_W / VB_W;
const sy = SLIDE_H / VB_H;
const EPS = 0.004;

function parseStyleStr(str) {
  const o = {};
  if (!str) return o;
  for (const part of str.split(';')) {
    const i = part.indexOf(':');
    if (i === -1) continue;
    o[part.slice(0, i).trim()] = part.slice(i + 1).trim();
  }
  return o;
}

function mergeStyle(parent, el) {
  return { ...parent, ...parseStyleStr(el.getAttribute('style') || '') };
}

function rgbToHex(v) {
  if (!v) return null;
  let m = v.match(/rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/i);
  if (m) {
    return [+m[1], +m[2], +m[3]]
      .map((n) => n.toString(16).padStart(2, '0'))
      .join('')
      .toUpperCase();
  }
  m = v.match(/#([0-9a-fA-F]{6})/);
  if (m) return m[1].toUpperCase();
  return null;
}

function parsePxPt(val, fallback) {
  if (val == null) return fallback;
  const m = String(val).match(/([\d.]+)\s*(px|pt)?/);
  if (!m) return fallback;
  return parseFloat(m[1]);
}

function parseOpacity(styleObj, el) {
  const o = styleObj.opacity != null ? parseFloat(styleObj.opacity) : NaN;
  if (!Number.isNaN(o)) return o;
  const a = el.getAttribute('opacity');
  if (a != null) return parseFloat(a);
  return 1;
}

function textContent(el) {
  let s = '';
  for (let c = el.firstChild; c; c = c.nextSibling) {
    if (c.nodeType === 3) s += c.data;
    else if (c.nodeType === 1) s += textContent(c);
  }
  return s
    .replace(/\u00a0/g, ' ')
    .replace(/\r?\n/g, '')
    .trim();
}

function parsePathPoints(d) {
  const raw = d.trim().split(/[\s,]+/).filter(Boolean);
  const pts = [];
  let i = 0;
  while (i < raw.length) {
    const t = raw[i];
    if (t === 'M' || t === 'L') {
      pts.push({ x: parseFloat(raw[i + 1]), y: parseFloat(raw[i + 2]) });
      i += 3;
    } else if (/^[\d.-]+$/.test(t) && pts.length && /^[\d.-]+$/.test(raw[i + 1])) {
      pts.push({ x: parseFloat(raw[i]), y: parseFloat(raw[i + 1]) });
      i += 2;
    } else i += 1;
  }
  return pts;
}

function walk(node, inherited, slide, pptx, emit) {
  if (!node || node.nodeType !== 1) return;
  const tag = String(node.tagName || '').toLowerCase();
  if (tag === 'defs' || tag === 'title' || tag === 'desc') return;

  const style = mergeStyle(inherited, node);

  if (tag === 'svg') {
    for (let c = node.firstChild; c; c = c.nextSibling) walk(c, style, slide, pptx, emit);
    return;
  }
  if (tag === 'g') {
    for (let c = node.firstChild; c; c = c.nextSibling) walk(c, style, slide, pptx, emit);
    return;
  }

  if (tag === 'rect') {
    emit.rect(node, style);
    return;
  }
  if (tag === 'line') {
    emit.line(node, style);
    return;
  }
  if (tag === 'path') {
    emit.path(node, style);
    return;
  }
  if (tag === 'text') {
    emit.text(node, style);
    return;
  }
}

function buildEmit(slide, pptx) {
  return {
    rect(el, st) {
      const x = parseFloat(el.getAttribute('x') || '0');
      const y = parseFloat(el.getAttribute('y') || '0');
      const w = parseFloat(el.getAttribute('width') || '0');
      const h = parseFloat(el.getAttribute('height') || '0');
      const rx = parseFloat(el.getAttribute('rx') || '0');
      const fillH = rgbToHex(st.fill);
      const strokeH = rgbToHex(st.stroke);
      let sw = parsePxPt(st['stroke-width'], 0.75);
      if (String(st['stroke-width'] || '').includes('px')) sw = Math.max(0.25, sw * 0.75);

      const op = parseOpacity(st, el);
      const fill =
        fillH != null
          ? { color: fillH, transparency: op < 1 ? Math.round((1 - op) * 100) : undefined }
          : { color: 'FFFFFF', transparency: 100 };

      const line =
        strokeH && sw > 0
          ? {
              color: strokeH,
              width: Math.max(0.25, sw),
              transparency: op < 1 ? Math.round((1 - op) * 100) : undefined,
            }
          : { width: 0 };

      const shape =
        rx > 0.5 ? pptx.ShapeType.roundRect : pptx.ShapeType.rect;
      const rectRadius =
        rx > 0.5 ? Math.min(0.5, rx / Math.min(w, h || 1)) : undefined;

      slide.addShape(shape, {
        x: x * sx,
        y: y * sy,
        w: Math.max(EPS, w * sx),
        h: Math.max(EPS, h * sy),
        fill,
        line,
        ...(rectRadius ? { rectRadius } : {}),
      });
    },

    line(el, st) {
      const x1 = parseFloat(el.getAttribute('x1'));
      const y1 = parseFloat(el.getAttribute('y1'));
      const x2 = parseFloat(el.getAttribute('x2'));
      const y2 = parseFloat(el.getAttribute('y2'));
      const strokeH = rgbToHex(st.stroke) || '73726C';
      let sw = parsePxPt(st['stroke-width'], 1.5);
      if (String(st['stroke-width'] || '').includes('px')) sw = Math.max(0.5, sw * 0.75);
      const op = parseOpacity(st, el);
      const marker = el.getAttribute('marker-end');
      const arrow = marker && marker.includes('arrow');

      const x1s = x1 * sx,
        y1s = y1 * sy,
        x2s = x2 * sx,
        y2s = y2 * sy;
      const minX = Math.min(x1s, x2s);
      const minY = Math.min(y1s, y2s);
      const w = Math.max(EPS, Math.abs(x2s - x1s));
      const h = Math.max(EPS, Math.abs(y2s - y1s));

      slide.addShape(pptx.ShapeType.line, {
        x: minX,
        y: minY,
        w,
        h,
        line: {
          color: strokeH,
          width: Math.max(0.5, sw),
          endArrowType: arrow ? 'arrow' : undefined,
          transparency: op < 1 ? Math.round((1 - op) * 100) : undefined,
        },
      });
    },

    path(el, st) {
      const d = el.getAttribute('d') || '';
      const pts = parsePathPoints(d);
      if (pts.length < 2) return;
      const strokeH = rgbToHex(st.stroke) || '73726C';
      let sw = parsePxPt(st['stroke-width'], 1);
      if (String(st['stroke-width'] || '').includes('px')) sw = Math.max(0.35, sw * 0.75);
      const op = parseOpacity(st, el);
      const dash = st['stroke-dasharray'] && !st['stroke-dasharray'].startsWith('none');
      const marker = el.getAttribute('marker-end');
      const arrow = marker && marker.includes('arrow');

      for (let i = 0; i < pts.length - 1; i++) {
        const x1 = pts[i].x,
          y1 = pts[i].y,
          x2 = pts[i + 1].x,
          y2 = pts[i + 1].y;
        const x1s = x1 * sx,
          y1s = y1 * sy,
          x2s = x2 * sx,
          y2s = y2 * sy;
        const minX = Math.min(x1s, x2s);
        const minY = Math.min(y1s, y2s);
        const w = Math.max(EPS, Math.abs(x2s - x1s));
        const h = Math.max(EPS, Math.abs(y2s - y1s));
        const last = i === pts.length - 2;
        slide.addShape(pptx.ShapeType.line, {
          x: minX,
          y: minY,
          w,
          h,
          line: {
            color: strokeH,
            width: Math.max(0.35, sw),
            dashType: dash ? 'dash' : 'solid',
            endArrowType: arrow && last ? 'arrow' : undefined,
            transparency: op < 1 ? Math.round((1 - op) * 100) : undefined,
          },
        });
      }
    },

    text(el, st) {
      const str = textContent(el);
      if (!str) return;

      const tx = parseFloat(el.getAttribute('x') || '0');
      const ty = parseFloat(el.getAttribute('y') || '0');
      const anchor = el.getAttribute('text-anchor') || st['text-anchor'] || 'start';
      const baseline =
        el.getAttribute('dominant-baseline') || st['dominant-baseline'] || 'auto';
      const transform = el.getAttribute('transform') || '';

      let fontSize = parsePxPt(st['font-size'], 12);
      const fillH = rgbToHex(st.fill) || '333333';
      const w600 = (st['font-weight'] || '').includes('500') || (st['font-weight'] || '').includes('600');
      const bold =
        (st['font-weight'] || '').includes('bold') ||
        (st['font-weight'] || '').includes('700') ||
        w600;
      const op = parseOpacity(st, el);

      const hIn = Math.max((fontSize * 1.35) / 72, 0.12);
      let wIn = Math.min(
        SLIDE_W * 0.94,
        Math.max(0.22, str.length * fontSize * 0.016)
      );
      if (str.length > 36) wIn = SLIDE_W * 0.9;

      let xIn = tx * sx;
      let yIn;
      if (baseline === 'central') yIn = ty * sy - hIn / 2;
      else yIn = ty * sy - (fontSize * 0.85) / 72;

      let align = 'left';
      if (anchor === 'middle') {
        align = 'center';
        xIn = tx * sx - wIn / 2;
      } else if (anchor === 'end') {
        align = 'right';
        xIn = tx * sx - wIn;
      }

      xIn = Math.max(0, Math.min(SLIDE_W - EPS - wIn, xIn));

      const rotMatch = transform.match(/rotate\s*\(\s*(-?[\d.]+)/);
      const rotate = rotMatch ? parseFloat(rotMatch[1]) : 0;

      slide.addText(str, {
        x: xIn,
        y: Math.max(0, yIn),
        w: wIn,
        h: hIn,
        fontSize,
        fontFace: 'Microsoft YaHei',
        color: fillH,
        bold,
        align,
        valign: 'middle',
        margin: 0,
        transparency: op < 1 ? Math.round((1 - op) * 100) : undefined,
        ...(rotate ? { rotate } : {}),
      });
    },
  };
}

function main() {
  const xml = fs.readFileSync(SVG_IN, 'utf8');
  const doc = new DOMParser({
    onError() {},
  }).parseFromString(xml, 'text/xml');

  const svg = doc.getElementsByTagName('svg')[0];
  if (!svg) throw new Error('No <svg> root');

  const pres = new pptxgen();
  pres.defineLayout({ name: 'FLOWCHART_PORTRAIT', width: SLIDE_W, height: SLIDE_H });
  pres.layout = 'FLOWCHART_PORTRAIT';
  pres.title = 'StrucLift 精确框架流程图';
  pres.author = 'StructLift';

  const slide = pres.addSlide();
  slide.background = { color: 'F8F8F8' };

  const emit = buildEmit(slide, pres);
  for (let c = svg.firstChild; c; c = c.nextSibling) walk(c, {}, slide, pres, emit);

  return pres.writeFile({ fileName: PPTX_OUT, compression: true });
}

main()
  .then(() => console.log('Wrote', PPTX_OUT))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
