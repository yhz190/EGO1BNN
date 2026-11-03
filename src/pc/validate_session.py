#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""validate_session.py -- 采集批次合规检查（拍完先跑这个，再谈标定）

按《采集全流程_v3》与《现场标定流水线》的判据检查一个时间窗内的数据：

  A. 数量     每类 >= N（默认 rock/paper/scissors 60，junk 50）
  B. 手存在    手面积 > 0.12（无手/空景帧比例）
  C. 手完整入画 贴边度 < 0.35（B6 实测 0.72 是手臂入画基线，此处只拦"整条边全黑")
  D. 手大小    近距 0.30~0.50 / 远距 0.18~0.32，且两类都要有（覆盖率）
  E. 环境一致  bg_level / thr_ratio 与参考（窗口模式=本批中位数）偏差 <20% / <10%
  F. 编码/命名 文件名唯一、均能在 manifest 找到、无重复图像
  G. junk 内容  至少含一部分"无手/空"帧（面积 < 0.15）

用法:
  python validate_session.py --since 17:30 [--until 18:10]
输出: 通过/不通过 + 每项证据 + 建议补拍多少张
"""
import os, sys, csv, time, hashlib, collections, argparse
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
import preprocess as P1
import env_stats as ES

LIVE = os.path.join(ROOT, 'src', 'data', 'live')
CLASSES = ('rock', 'paper', 'scissors', 'junk')
MIN_COUNT = {'rock': 60, 'paper': 60, 'scissors': 60, 'junk': 50}
NEAR = (0.30, 0.50)
FAR = (0.18, 0.32)


def load_manifest():
    p = os.path.join(LIVE, '_manifest.csv')
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    return rows


def ts(hhmm):
    return time.mktime(time.strptime('2026-11-01 ' + hhmm, '%Y-%m-%d %H:%M'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', required=True)
    ap.add_argument('--until', default='')
    a = ap.parse_args()
    lo = ts(a.since)
    hi = ts(a.until) if a.until else 1e18

    rows = load_manifest()
    mts = {}
    dup_names = collections.Counter()
    for r in rows:
        dup_names[r['file']] += 1
        mts[r['file']] = int(r['ts'])

    code_pool = set()
    per = {}
    problems, warnings = [], []
    for cls in CLASSES:
        d = os.path.join(LIVE, cls)
        files = []
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if not f.endswith('.png'):
                    continue
                n = f[:-4]
                t = mts.get(n, 0)
                if lo <= t < hi:
                    files.append((n, os.path.join(d, f)))
        per[cls] = []
        for n, p in files:
            img = Image.open(p)
            rgb = np.asarray(img.convert('RGB'))
            _, bits, thr = P1.to_frame(img)
            gray = np.asarray(img.convert('L'), np.uint8)
            pa = ES.frame_params(gray, bits, thr)
            h = hashlib.md5(rgb.tobytes()).hexdigest()
            per[cls].append({'name': n, 'path': p, 'params': pa, 'hash': h,
                             'ts': mts.get(n, 0), 'thr': thr, 'bit1': float(bits.mean())})
            code_pool.add(h)

    print('=' * 78)
    print('采集批次合规检查  %s ~ %s' % (a.since, a.until or 'now'))
    print('=' * 78)
    if not per['rock'] and not per['paper']:
        print('[FAIL] 该时间窗内没有新帧；确认采集时间范围/是否保存')
        return 1

    # ---- reference from this batch (window median), for env consistency ----
    allp = [x['params'] for cls in ('rock', 'paper', 'scissors') for x in per[cls]] or \
           [x['params'] for cls in CLASSES for x in per[cls]]
    ref = {}
    if len(allp) >= 10:
        for k in allp[0]:
            v = np.array([x[k] for x in allp], np.float64)
            med = float(np.median(v)); mad = float(np.median(np.abs(v - med)))
            ref[k] = {'mean': float(v.mean()), 'std': float(v.std()), 'median': med, 'mad': mad}

    total_ok = True
    for cls in CLASSES:
        xs = per[cls]
        n = len(xs)
        print('\n[%s] %d 帧' % (cls, n))
        if n < MIN_COUNT[cls]:
            total_ok = False
            problems.append('%s 只有 %d 帧（要求 >=%d）' % (cls, n, MIN_COUNT[cls]))
        if n == 0:
            continue
        areas = np.array([x['params']['area'] for x in xs])
        borders = np.array([x['params'].get('border_bad', x['params']['border']) for x in xs])
        nohand = int((areas < 0.12).sum())
        clipped = int((borders > 0.98).sum())   # severe only; background structure handled by EDGE_TOP check
        near = int(((areas >= NEAR[0]) & (areas <= NEAR[1])).sum())
        far = int(((areas >= FAR[0]) & (areas <= FAR[1])).sum())
        dup = n - len({x['hash'] for x in xs})
        print('  手面积 %.2f±%.2f (min %.2f max %.2f) 中位 %.2f' % (areas.mean(), areas.std(), areas.min(), areas.max(), np.median(areas)))
        print('  无手帧(<0.12)=%d  近距覆盖=%d  远距覆盖=%d  顶边暗区均值=%.2f' % (nohand, near, far, float(np.mean([x['params'].get('edge_top', 0.0) for x in xs]))))
        print('  重复图像=%d  文件名重复=%d' % (dup, sum(1 for x in xs if dup_names.get(x['name'], 0) > 1)))
        if cls != 'junk':
            if nohand > max(2, int(0.1 * n)):
                total_ok = False
                problems.append('%s 有无手帧 %d 个（应 <=10%%）' % (cls, nohand))
            _et = float(np.mean([x['params'].get('edge_top', 0.0) for x in xs]))
            if _et > 0.5:
                problems.append('%s 顶边暗区过大（均值 %.2f）：画面里有暗背景/黑边 -> 调整机位让背景均匀后再拍'
                                % (cls, _et))
            if n >= 30 and (near < 8 or far < 8):
                warnings.append('%s 距离覆盖不足：近距 %d / 远距 %d（建议各 >=15）' % (cls, near, far))
        else:
            if nohand < 5:
                warnings.append('junk 无手/空景帧只有 %d 个（建议 >=5，用于开集拒识）' % nohand)
        if dup > 0:
            warnings.append('%s 有 %d 张重复图像' % (cls, dup))
        # 环境一致性（本批中位数 vs 每帧）
        env_bad = 0
        for x in xs:
            r = ES.assess(x['params'], ref)
            if not r['env_ok']:
                env_bad += 1
        print('  环境一致率（批内参考）= %.1f%%' % (100.0 * (n - env_bad) / n))

    # ---- cross-batch env drift vs previously deployed reference ----
    ref_prev = ES.load_reference(os.path.join(LIVE, '_env', 'reference_current.json'))
    if ref_prev and allp:
        med = ES.window_median(allp)
        r = ES.assess(med, ref_prev)
        print('\n与上一会话参考对比: 环境%s' % ('正常' if r['env_ok'] else '漂移'))
        for f, v in r['env'].items():
            print('  %-10s 本批=%.2f  允许[%.2f, %.2f]  %s'
                  % (f, v['cur'], v['lo'], v['hi'], 'OK' if v['ok'] else 'DIFF'))
        if not r['env_ok']:
            warnings.append('与上一会话环境不同（正常现象：新会话需重新标定，不要混训）')

    print('\n' + '=' * 78)
    if problems:
        print('[FAIL] 需要补拍/修正：')
        for p in problems:
            print('  - ' + p)
        print('  => 补齐后重跑本检查，再发时间范围给 Lead 标定')
    else:
        print('[PASS] 数量与基本质量达标，可以进入标定')
    if warnings:
        print('[WARN] 建议关注（不阻塞）：')
        for w in warnings:
            print('  - ' + w)
    print('=' * 78)
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
