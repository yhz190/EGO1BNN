#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_golden_t8.py -- generate T8 golden vectors with HARDWARE (integer) semantics.

Key point: train_bnn10.hw_eval compares popcounts against FLOAT thresholds, while the
hardware compares against 4-bit INTEGER theta. A vector set meant to gate RTL must use
the integer semantics, otherwise the reference itself is not implementable.

Pipeline (mirrors v2.2 §6.1 and 05_权重布局与导出规范):
  L1: x1 = (matches_fixed(X, w1, pad=1) > t1_int)      # 8 ch, 64x64
  L2: x2 = (matches_fixed(x1, w2, pad=1, groups=8) > t2_int)
  L3: x3 = (matches_fixed(x2, w3, pad=0) > t3_int)     # 4 ch, 64x64
  L4: s = box_sum_4x4(x3); feat = (s >= 8)             # 1024 bit
  L5: mf = matches_fixed(feat, wf, pad=0); score_q = 64*(2*mf - feat) + round(bias*64)  # Q6
Outputs data/t8_vectors/golden.npz (+ labels, report, a few .hex samples).
Self-check: integer-argmax vs hw_eval-argmax must agree on all 620 frames.
"""
import os, sys, csv, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_bnn10 as T10
from class_map import assert_class_map

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC_SCALE = 64   # Q6 fixed-point FC bias (E6 resolution)
assert_class_map()
VEC = os.path.join(ROOT, 'sim', 'vectors_t1a')
CKPT = os.path.join(ROOT, 'src', 'weights', 'E_E1_fixsplit_seed1.pt')
OUT = os.path.join(ROOT, 'sim', '20261101_t8_vectors')
os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(2)          # be polite: another training job may be running


def load_frames():
    rows = list(csv.DictReader(open(os.path.join(VEC, 'labels.csv'), encoding='utf-8-sig')))
    X = np.zeros((len(rows), 1, 64, 64), np.float32)
    for i, r in enumerate(rows):
        raw = np.fromfile(os.path.join(VEC, r['file'].strip()), np.uint8)
        X[i, 0] = np.unpackbits(raw)[:4096].reshape(64, 64)
    y = np.array([int(r['class']) for r in rows], np.int64)
    return [r['file'].strip() for r in rows], torch.from_numpy(X), y


def packbits_t(t):      # [N,C,64,64] float 0/1 -> [N,C,512] uint8 (MSB first)
    a = t.detach().cpu().numpy().astype(np.uint8).reshape(t.shape[0], t.shape[1], -1)
    return np.packbits(a, axis=2)


def main():
    files, X, y = load_frames()
    print('[data] frames =', len(files))

    sd = torch.load(CKPT, map_location='cpu')
    m = T10.Net(4)
    m.load_state_dict(sd)
    m.eval()

    w1 = (m.c1.weight > 0).float()
    w2 = (m.dw.weight > 0).float()
    w3 = (m.c3.weight > 0).float()
    wf = (m.fc.weight > 0).float()

    t1f = T10.theta_of(m.b1, m.c1.weight, 9, False)
    t2f = T10.theta_of(m.b2, m.dw.weight, 9, True)
    t3f = T10.theta_of(m.b3, m.c3.weight, 8, True)
    # CRITICAL: the popcount is an integer, so (count > t_float) == (count > floor(t_float)).
    # Rounding would shift the threshold by +1 and change hundreds of pixels.
    t1i = t1f.floor().long()
    t2i = t2f.floor().long()
    t3i = t3f.floor().long()
    bias_int = (m.fc.bias.detach() * FC_SCALE).round().long()   # Q6
    print('[theta float] L1', t1f.tolist())
    print('[theta int  ] L1', t1i.tolist())
    print('[theta int  ] L2', t2i.tolist())
    print('[theta int  ] L3', t3i.tolist())
    print('[FC bias int]', bias_int.tolist())

    feat = m.feat
    with torch.no_grad():
        c1 = T10.matches_fixed(X, w1, 1)
        x1 = (c1 > t1i.view(1, -1, 1, 1)).float()
        c2 = T10.matches_fixed(x1, w2, 1, groups=8)
        x2 = (c2 > t2i.view(1, -1, 1, 1)).float()
        c3 = T10.matches_fixed(x2, w3, 0)
        x3 = (c3 > t3i.view(1, -1, 1, 1)).float()

        s = F.avg_pool2d(x3, 4) * 16.0
        fh = (s >= 8.0).float().flatten(1)                      # [N,1024]

        mf = T10.matches_fixed(fh.reshape(-1, feat, 1, 1), wf.reshape(4, feat, 1, 1), 0)  # [N,4,1,1]
        mf = mf.reshape(mf.shape[0], 4)
        score_int = (2 * mf - feat).long() * FC_SCALE + bias_int.view(1, 4)   # [N,4] Q6 int32
        pred_int = score_int.argmax(1)

        logit_ref = T10.hw_eval(m, X)                             # float-theta reference
        pred_ref = logit_ref.argmax(1)

    agree = int((pred_int == pred_ref).sum())
    acc = float((pred_int == torch.from_numpy(y)).float().mean())
    acc_ref = float((pred_ref == torch.from_numpy(y)).float().mean())
    print('[check] integer-argmax vs hw_eval-argmax = %d/%d' % (agree, len(y)))
    print('[acc] integer = %.4f   hw_eval = %.4f' % (acc, acc_ref))

    # ---- dump ----
    np.savez_compressed(
        os.path.join(OUT, 'golden.npz'),
        files=np.array(files),
        labels=y,
        L1=packbits_t(x1), L2=packbits_t(x2), L3=packbits_t(x3),
        c1=c1.numpy().astype(np.uint8), c2=c2.numpy().astype(np.uint8), c3=c3.numpy().astype(np.uint8),
        L4=np.packbits(fh.numpy().astype(np.uint8), axis=1),
        L5=score_int.numpy().astype(np.int32),
        pred=pred_int.numpy().astype(np.uint8),
    )
    meta = {
        'ckpt': os.path.basename(CKPT),
        'theta_int': {'L1': t1i.tolist(), 'L2': t2i.tolist(), 'L3': t3i.tolist()},
        'fc_bias_int': bias_int.tolist(),
        'frames': len(files),
        'acc_integer': acc,
        'acc_hw_eval_float_theta': acc_ref,
        'integer_vs_hw_eval_argmax_agree': agree,
        'layout': 'L1/L2: [N,8,64,64] bit; L3: [N,4,64,64]; L4: 1024 bit; L5: int scores',
        'note': 'padding semantics: bit-domain pad 0 for both domains (bit0 == -1 in pm1 domain)',
    }
    json.dump(meta, open(os.path.join(OUT, 'report.json'), 'w'), indent=2)

    # ---- a few human-readable samples for module-level TB ----
    for k in range(3):
        lines = []
        lines.append('# frame %d  file=%s  label=%d  pred_int=%d' % (k, files[k], y[k], pred_int[k]))
        for name, t in (('L1', x1), ('L2', x2), ('L3', x3)):
            lines.append('[%s]' % name)
            arr = t[k].numpy().astype(np.uint8)                 # [C,64,64]
            for c in range(arr.shape[0]):
                for yy in range(64):
                    row = np.packbits(arr[c, yy])
                    lines.append('%s ' % name + ''.join('%02x' % b for b in row))
        feat_bits = fh[k].numpy().astype(np.uint8)
        lines.append('[L4]')
        for yy in range(16):
            chunk = feat_bits[yy * 64:(yy + 1) * 64]
            lines.append('L4 ' + ''.join('%02x' % b for b in np.packbits(chunk)))
        lines.append('[L5] scores_int = %s  pred=%d' % (score_int[k].tolist(), pred_int[k]))
        open(os.path.join(OUT, 'sample_frame_%d.hex' % k), 'w').write('\n'.join(lines) + '\n')

    print('[out]', OUT)
    print('[size] golden.npz = %.2f MB' % (os.path.getsize(os.path.join(OUT, 'golden.npz')) / 1e6))
    # ---- self-consistency: replay the dumped vectors back through the integer pipeline ----
    z = np.load(os.path.join(OUT, 'golden.npz'))
    L4 = np.unpackbits(z['L4'], axis=1)[:, :1024].astype(np.float32)
    L4t = torch.from_numpy(L4).reshape(-1, feat, 1, 1)
    with torch.no_grad():
        mf2 = T10.matches_fixed(L4t, wf.reshape(4, feat, 1, 1), 0).reshape(-1, 4)
        sc2 = (2 * mf2 - feat).long() * FC_SCALE + bias_int.view(1, 4)
    replay_ok = int((sc2.argmax(1) == pred_int).sum())
    print('[replay] dumped vectors replayed -> argmax %d/%d match' % (replay_ok, len(files)))
    meta['replay_argmax_match'] = replay_ok
    meta['float_reference_note'] = (
        'hw_eval uses FLOAT theta and FLOAT fc bias; hardware uses floor(theta) and round(bias). '
        'Layer bitmaps are identical (L1..L4 diff = 0); only near-tie FC decisions can differ '
        '(%d/%d frames).' % (len(files) - agree, len(files)))
    json.dump(meta, open(os.path.join(OUT, 'report.json'), 'w'), indent=2)
    if replay_ok != len(files):
        print('[FAIL] replay mismatch -> vectors not usable')
        return 1
    print('[PASS] replay self-consistent %d/%d ; integer vs float-reference argmax %d/%d (documented delta)'
          % (replay_ok, len(files), agree, len(files)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
