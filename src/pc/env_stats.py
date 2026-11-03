#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""env_stats.py v3 -- pose-robust ENV metrics + specific-field diagnostics for the capture UI.

v2 bug (found 2026-11-01 17:35): ENV used whole-frame brightness and Otsu threshold, both of
which move when the hand changes pose -> "changing gesture turns the environment red".
v3 computes ENV on BACKGROUND pixels only (bits==1), so hand pose cannot move it:

  ENV fields  (must stay inside the band; lighting/camera/background)
      bg_level  = median gray of background pixels        (~background brightness)
      bg_std    = std of background pixels                (~background texture)
      thr_ratio = otsu_threshold / bg_level               (~lighting-normalised contrast)
  POSE fields (free to vary; hint only)
      area, centroid, border

The UI renders these through PIL (Chinese-capable); each failing field is named with its value
and allowed band instead of a bare red mark.
"""
import os, sys, json
import numpy as np
from PIL import Image

# ---- control bands ---------------------------------------------------------------
ENV_FIELDS = ('edge_top', 'edge_lr')      # background structure: pose-independent
INFO_FIELDS = ('bg_level', 'bg_std', 'thr_ratio')
ENV_ABS_TOL = {'edge_top': 0.15, 'edge_lr': 0.15, 'bg_level': 8.0, 'bg_std': 6.0, 'thr_ratio': 0.05}
ENV_REL_FRAC = {'bg_level': 0.20, 'bg_std': 0.30, 'thr_ratio': 0.10}
# upper bound for the sigma-widened ENV band: natural pose variation must not widen it forever
ENV_TOL_CAP = {'bg_level': 30.0, 'bg_std': 30.0, 'thr_ratio': 0.15}
POSE_FIELDS = ('area', 'centroid', 'border')
POSE_ABS_TOL = {'area': 0.10, 'centroid': 0.10, 'border': 0.15}
K_SIGMA = 3.0

ENV_LABEL = {'bg_level': '背景亮度', 'bg_std': '背景纹理', 'thr_ratio': '光照对比',
             'edge_top': '顶边暗区', 'edge_lr': '侧边暗区'}
POSE_LABEL = {'area': '手面积', 'centroid': '偏离中心', 'border': '贴边'}

HINTS = {
    'edge_top': '顶边出现暗背景（天花板/黑边/阴影）→ 相机下压一点或补光，让画面全是亮背景',
    'edge_lr': '侧边出现暗背景 → 调整机位/补光，让画面全是亮背景',
    'bg_level': '背景亮度变了 → 恢复灯光/曝光（环境项必须稳定）',
    'bg_std': '背景纹理变了 → 清走桌上杂物/检查镜头',
    'thr_ratio': '光照对比变了 → 恢复原来灯光，避免逆光/阴影',
    'area_small': '手偏小（提示，不拦截）→ 可靠近一点',
    'area_big': '手偏大（提示，不拦截）→ 可退后一点',
    'centroid': '手偏离画面中心（提示，不拦截）',
    'border': '手/臂出框（提示，不拦截）→ 让手完整入画',
}


def frame_params(gray_u8, bits, thr):
    """pose-robust environment fingerprint + pose parameters of one frame"""
    hb, wb = bits.shape
    gg = np.asarray(Image.fromarray(gray_u8).resize((wb, hb), Image.BILINEAR), np.float32)
    bright = gg[bits > 0]
    if bright.size < 50:
        bright = gg.reshape(-1)
    bg_level = float(np.median(bright))
    bg_std = float(bright.std())
    dark = (1 - bits).astype(np.float32)
    ys, xs = np.nonzero(dark)
    cy = float(ys.mean()) / dark.shape[0] if len(ys) else 0.5
    cx = float(xs.mean()) / dark.shape[1] if len(xs) else 0.5
    b = 3
    edge_top = float(dark[:b, :].mean())
    edge_bottom = float(dark[-b:, :].mean())      # arm usually enters here -> NOT a defect
    edge_lr = float(max(dark[:, :b].mean(), dark[:, -b:].mean()))
    border = float(max(edge_top, edge_bottom, edge_lr))   # kept for compatibility/reference
    border_bad = float(max(edge_top, edge_lr))            # gesture touching top/left/right = clipped
    return {
        'bg_level': bg_level, 'bg_std': bg_std,
        'thr_ratio': float(thr) / max(bg_level, 1.0),
        'area': float(dark.mean()), 'cy': cy, 'cx': cx,
        'centroid': float(((cy - 0.5) ** 2 + (cx - 0.5) ** 2) ** 0.5),
        'border': border, 'border_bad': border_bad,
        'edge_top': edge_top, 'edge_bottom': edge_bottom, 'edge_lr': edge_lr,
        'thr': float(thr),
        'bit1': float(bits.mean()), 'mean': float(gg.mean()),
    }


def session_stats(paths, preprocess_to_frame, limit=None):
    acc = []
    for i, p in enumerate(paths):
        if limit and i >= limit:
            break
        img = Image.open(p)
        _, bits, thr = preprocess_to_frame(img)
        gray = np.asarray(img.convert('L'), np.uint8)
        acc.append(frame_params(gray, bits, thr))
    if not acc:
        raise RuntimeError('no frames')
    out = {}
    for k in acc[0]:
        v = np.array([a[k] for a in acc], np.float64)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        out[k] = {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                  'median': med, 'mad': mad}
    return out


def band(field, ref_field):
    """(lo, hi) control band. Centred on the median; width from the robust MAD sigma
    (falls back to the plain std for old references), never narrower than the absolute floor."""
    m = ref_field.get('median', ref_field['mean'])
    sigma = ref_field.get('mad', 0.0) * 1.4826
    if sigma <= 0:
        sigma = ref_field.get('std', 0.0)
    abs_floor = ENV_ABS_TOL.get(field, 0.0) or POSE_ABS_TOL.get(field, 0.0)
    tol = max(abs_floor, K_SIGMA * sigma)
    if field in ENV_REL_FRAC:                     # ENV uses a relative band: robust in uneven scenes
        tol = max(abs_floor, ENV_REL_FRAC[field] * abs(m))
    if field in ENV_TOL_CAP:
        tol = min(tol, ENV_TOL_CAP[field])
    return m - tol, m + tol


def assess(cur, ref):
    """per-field verdicts; pose never blocks capture"""
    env_items, pose_items, hints = {}, {}, []
    for f in ENV_FIELDS:
        if f not in ref or f not in cur:
            continue
        lo, hi = band(f, ref[f])
        ok = lo <= cur[f] <= hi
        env_items[f] = {'cur': cur[f], 'lo': lo, 'hi': hi, 'ok': ok}
        if not ok:
            hints.append('%s: %s=%.3f 超出 [%.3f, %.3f] → %s'
                         % (ENV_LABEL[f], f, cur[f], lo, hi, HINTS[f]))
    for f in POSE_FIELDS:
        if f not in ref or f not in cur:
            continue
        lo, hi = band(f, ref[f])
        ok = lo <= cur[f] <= hi
        pose_items[f] = {'cur': cur[f], 'lo': lo, 'hi': hi, 'ok': ok}
        if not ok:
            if f == 'area':
                hints.append('手面积=%.2f 超出 [%.2f, %.2f]（提示，不拦截）'
                             % (cur[f], lo, hi))
            else:
                hints.append('%s: %s=%.2f 超出 [%.2f, %.2f]（提示，不拦截）'
                             % (POSE_LABEL[f], f, cur[f], lo, hi))
    return {'env_ok': all(v['ok'] for v in env_items.values()), 'env': env_items,
            'pose_ok': all(v['ok'] for v in pose_items.values()), 'pose': pose_items,
            'hints': hints}


def status_lines(cur, ref, res):
    """ready-to-draw UI lines: list of (text, kind) where kind in {'ok','warn','bad','info','hint'}"""
    lines = []
    if res is None or not res['env']:
        lines.append(('环境未锁定：按 S 锁定（先做 3~4 个同类手势）', 'warn'))
        lines.append(('手面积=%.2f  偏离中心=%.2f  贴边=%.2f'
                      % (cur.get('area', 0), cur.get('centroid', 0), cur.get('border', 0)), 'info'))
        return lines
    if len(res['env']) < len(ENV_FIELDS):
        lines.append(('参考是旧版（缺字段）→ 请按 S 重新锁定环境', 'warn'))
        lines.append(('手面积=%.2f  偏离中心=%.2f  贴边=%.2f'
                      % (cur['area'], cur['centroid'], cur['border']), 'info'))
        return lines
    env = res['env']
    detail = []
    for f, label in (('edge_top', '顶边暗区'), ('edge_lr', '侧边暗区')):
        if f not in env:
            continue
        v = env[f]
        mark = '[OK]' if v['ok'] else '[X]'
        refv = ref.get(f, {}).get('median', ref.get(f, {}).get('mean', float('nan')))
        detail.append('%s%s=%.2f(参考%.2f, 允许%.2f~%.2f)'
                      % (mark, label, v['cur'], refv, v['lo'], v['hi']))
    lines.append(('环境 ' + ('正常' if res['env_ok'] else '漂移') + ' | ' + '  '.join(detail),
                  'ok' if res['env_ok'] else 'bad'))
    pose = res['pose']
    pl = []
    for f, label in (('area', '手面积'), ('centroid', '偏离中心'), ('border', '贴边')):
        if f in pose:
            pl.append('%s=%.2f' % (label, pose[f]['cur']))
    if ref and 'bg_std' in ref:
        lines.append(('信息: 背景亮度=%.0f(参考%.0f)  光照对比=%.2f(参考%.2f)  手面积=%.2f  偏离中心=%.2f'
                      % (cur.get('bg_level', 0), ref['bg_level'].get('median', 0),
                         cur.get('thr_ratio', 0), ref['thr_ratio'].get('median', 0),
                         cur.get('area', 0), cur.get('centroid', 0)), 'info'))
    else:
        lines.append(('姿态（自由） | ' + '  '.join(pl), 'info'))
    if not res['env_ok']:
        lines.append(('参考可能来自旧会话，或光照已变 → 保持现在的光照，按 S 重新锁定（C 可清除参考）', 'warn'))
    for h in res['hints'][:2]:
        lines.append((h, 'bad' if not res['env_ok'] else 'hint'))
    return lines


def window_median(hist, fields=None):
    """median of the last N frame params -> the value the UI judges (stable, flicker-free)"""
    if not hist:
        return {}
    keys = fields or list(hist[0].keys())
    return {k: float(np.median([h[k] for h in hist])) for k in keys if k in hist[0]}


def window_spread(hist):
    """relative spread (IQR/median) of the window: shows if the scene is moving"""
    if len(hist) < 3:
        return {}
    out = {}
    for k in ('bg_level', 'thr_ratio', 'area'):
        if k not in hist[0]:
            continue
        v = np.array([h[k] for h in hist], np.float64)
        med = float(np.median(v)) or 1e-6
        out[k] = float((np.percentile(v, 75) - np.percentile(v, 25)) / abs(med))
    return out


def stability(recent, ref=None):
    if len(recent) < 3:
        return {}
    return {f: float(np.std([r[f] for r in recent])) for f in ('area', 'centroid', 'bg_level')}


def load_reference(path):
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None


def save_reference(stats, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(stats, open(path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


# ---- CLI -------------------------------------------------------------------------
def main():
    import argparse, csv, time
    sys.path.insert(0, os.path.dirname(__file__))
    import preprocess as P1
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LIVE = os.path.join(ROOT, 'src', 'data', 'live')
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='')
    ap.add_argument('--until', default='')
    ap.add_argument('--save', default='')
    ap.add_argument('--compare', default='')
    ap.add_argument('--limit', type=int, default=60)
    a = ap.parse_args()
    man = {}
    mp = os.path.join(LIVE, '_manifest.csv')
    if os.path.exists(mp):
        man = {r['file']: int(r['ts']) for r in csv.DictReader(open(mp, encoding='utf-8'))}

    def ts(x):
        return time.mktime(time.strptime('2026-11-01 ' + x, '%Y-%m-%d %H:%M')) if x else None

    lo, hi = ts(a.since), ts(a.until)
    paths = []
    for cls in ('rock', 'paper', 'scissors', 'junk'):
        d = os.path.join(LIVE, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(x for x in os.listdir(d) if x.endswith('.png')):
            t = man.get(f[:-4])
            if t is not None:
                if lo is not None and t < lo:
                    continue
                if hi is not None and t >= hi:
                    continue
            paths.append(os.path.join(d, f))
    paths = paths[:a.limit]
    st = session_stats(paths, P1.to_frame, limit=a.limit)
    print('  frames=%d (v3 pose-robust ENV)' % min(len(paths), a.limit))
    for k in ENV_FIELDS + POSE_FIELDS:
        print('    %-10s mean=%8.3f std=%.4f' % (k, st[k]['mean'], st[k]['std']))
    if a.save:
        save_reference(st, a.save)
        print('  saved ->', a.save)
    if a.compare:
        ref = load_reference(a.compare)
        if ref is None:
            print('[FAIL] reference missing')
            return 1
        cur = {k: v['mean'] for k, v in st.items()}
        res = assess(cur, ref)
        for text, kind in status_lines(cur, ref, res):
            print('  [%s] %s' % (kind, text))
        print('  环境正常 =', res['env_ok'], '（姿态项从不拦截）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
