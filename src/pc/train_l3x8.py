#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_l3x8.py -- v2.3 容量阶梯 L3：输出通道 4 -> 8（pool=4，特征 1024 -> 2048）。

唯一变量 = L3 输出通道数（c3: 8->4  ->  c3: 8->8）。其余尽量与 `train_fix.py` 的
E1 基线（同域 held-out 92.57%，seed1 / 30ep / teacher init，3 seed 89.9±3.2）对齐：

  * 数据划分：直接 import train_fix.build_split(seed=7, holdout=0.2)
      - 先按**原始文件** 80/20 划分，再只对训练集做水平翻转增广；
      - junk 训练/测试分源（train junk 只用 train 图，test junk 只用 held-out 图）；
      - 实测训练 6432 / 测试 1104（断言）。
  * 训练配方：Adam lr=2e-4 + CosineAnnealing(T_max=epochs)、bs=128、恒等 STE、
      BN 前置、w 不裁剪、无 EMA（E1 原配方）。
  * teacher 初始化：weights/float_honest.pt（train_honest.BNNet）。
      c1/dw/c3 权重 + b1/b2/b3 全量复制；**fc 默认随机**（train_fix.run_one 就是只复制
      卷积与 BN，不复制 fc）；--fc-init teacher 时才按层平铺复制，作对照。
  * 评测口径：硬件精确整数前向 hw_eval8()（权重 sign、L1/L2/L3 各自每通道 θ 由 BN fold、
      FC 整数偏置 Q8.8），并与 train_bnn10.hw_eval 交叉校验。

硬门（每次评估都打印 agree；只有 float==exact 逐点相同的评估才允许被选为 best，
最终被选中的模型重新加载后必须再次通过严格断言）：

    float argmax == exact argmax 逐点相同 (agree == N)

用法（单进程）：
  python src/pc/train_l3x8.py --init teacher --epochs 40 \
         --out board/weights/L3x8_best.pt
  python src/pc/train_l3x8.py --verify board/weights/L3x8_best.pt
  python src/pc/train_l3x8.py --verify ... --theta-quant floor
"""
import os
import sys
import copy
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CPU 纪律：脚本开头固定 4 线程（单进程，禁止并行 sweep） ---
torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bnn2 import load_bin, junk_from_images          # noqa: E402  与基线同一份 IO
from train_bnn5 import SignSTE                             # noqa: E402  恒等 STE
import train_honest                                        # noqa: E402  teacher 结构 BNNet
import train_bnn10 as T10                                  # noqa: E402  交叉校验用参考 hw_eval
import train_fix                                           # noqa: E402  基线数据划分（完全相同）

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = os.path.join(ROOT, 'src', 'weights')
TEACHER = os.path.join(W, 'float_honest.pt')
CLASSES = ['rock', 'paper', 'scissors', 'junk']
N_OUT = 8          # L3 输出通道数（本次实验唯一变量）
POOL = 4           # 与基线相同
SCALE_FC = 64     # FC 整数偏置定点标度（Q6，E6 定案：Q6 在 6 ckpt 上 100% 与 float 一致；Q8.8 反例 1 个）


# ------------------------------------------------------------------ 模型（复制 train_bnn10.Net 并扩展 c3 4->cout3）
class Net8(nn.Module):
    """train_bnn10.Net 的 L3=cout3 通道版：forward 逐行相同，仅 c3/b3 输出通道 4->cout3。"""

    def __init__(self, pool=POOL, cout3=N_OUT):
        super().__init__()
        self.pool = pool
        self.cout3 = cout3
        self.feat = cout3 * (64 // pool) ** 2
        self.c1 = nn.Conv2d(1, 8, 3, padding=0, bias=False)
        self.b1 = nn.BatchNorm2d(8)
        self.dw = nn.Conv2d(8, 8, 3, padding=0, groups=8, bias=False)
        self.b2 = nn.BatchNorm2d(8)
        self.c3 = nn.Conv2d(8, cout3, 1, bias=False)
        self.b3 = nn.BatchNorm2d(cout3)
        self.fc = nn.Linear(self.feat, 4, bias=True)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        x = SignSTE.apply(self.b1(F.conv2d(x, SignSTE.apply(self.c1.weight))))
        x = F.pad(x, (1, 1, 1, 1), value=-1.0)
        x = SignSTE.apply(self.b2(F.conv2d(x, SignSTE.apply(self.dw.weight), groups=8)))
        x = SignSTE.apply(self.b3(F.conv2d(x, SignSTE.apply(self.c3.weight))))
        s = F.avg_pool2d(x, self.pool) * (self.pool ** 2)
        x = SignSTE.apply(s + 1e-6)
        return F.linear(x.flatten(1), SignSTE.apply(self.fc.weight), self.fc.bias)


# ------------------------------------------------------------------ 硬件精确整数前向
def matches_fixed(x, w, pad=1, groups=1):
    """与 train_bnn10.matches_fixed 完全一致（0/1 域 XNOR 等价计数）。"""
    m1 = F.conv2d(F.pad(x, (pad,) * 4, value=0.0), w, None, 1, 0, groups=groups)
    m2 = F.conv2d(F.pad(1 - x, (pad,) * 4, value=1.0), 1 - w, None, 1, 0, groups=groups)
    return m1 + m2


def bn_c(bn):
    """BN fold：θ 的实值中心 c = μ − β·σ/γ（与 train_bnn10.bn_c 一致）。"""
    g = bn.weight.detach().clamp(min=1e-6)
    b = bn.bias.detach()
    mu = bn.running_mean.detach()
    sd = (bn.running_var.detach() + bn.eps).sqrt()
    return mu - b * sd / g


def theta_of(bn, w, K, pm1):
    """折叠成 popcount 整数域阈值（与 train_bnn10.theta_of 一致）。"""
    c = bn_c(bn)
    if pm1:
        return (K + c) / 2.0
    s = w.detach().sign().sum(dim=(1, 2, 3))
    return K / 2.0 - s / 2.0 + c


def quant_theta(t, mode='none'):
    """θ 的硬件存储口径。

    none  : float θ（软件参考，最精确）
    floor : 只存整数部分 floor(θ)——**与 float θ 逐点严格等价**：卷积累加值是整数，
            对任意实数 θ 有 (acc > θ) <=> (acc > floor(θ))，故 0 位小数即可。
    u8    : 8 bit + 偏移 128 + 步长 1/8（误差≤1/16）。**故意保留的反例**：阈值跨过整数时
            该通道判决整段翻转，精度会崩（见实验报告），说明"照搬 1/16 定点"是错的。
    """
    if mode == 'none':
        return t
    if mode == 'floor':
        return torch.floor(t)
    if mode == 'u8':
        q = torch.clamp(torch.round(t * 8.0 + 128.0), 0.0, 255.0)
        return (q - 128.0) / 8.0
    raise ValueError(mode)


def hw_eval8(m, X, theta_quant='floor', fc_scale=SCALE_FC):
    """硬件精确整数前向（0/1 域）：权重 sign + L1/L2/L3 每通道 θ + FC 整数偏置。

    返回 (ex_logit, ex_int)：
      ex_logit  实值域得分（bias 已量化到 1/fc_scale 后换回实值），用于与 float 比较 maxdiff；
      ex_int    纯整数得分 fc_scale*(2*mf-feat) + bias_q，硬件里就是它做 argmax。
    """
    w1 = (m.c1.weight > 0).float()
    w2 = (m.dw.weight > 0).float()
    w3 = (m.c3.weight > 0).float()
    wf = (m.fc.weight > 0).float()
    t1 = quant_theta(theta_of(m.b1, m.c1.weight, 9, False), theta_quant)   # L1: 3x3x1, K=9
    t2 = quant_theta(theta_of(m.b2, m.dw.weight, 9, True), theta_quant)    # L2: 3x3 dw, K=9
    t3 = quant_theta(theta_of(m.b3, m.c3.weight, 8, True), theta_quant)    # L3: 1x1 8->8, K=8
    x1 = (matches_fixed(X, w1, 1) > t1.view(1, -1, 1, 1)).float()
    x2 = (matches_fixed(x1, w2, 1, groups=8) > t2.view(1, -1, 1, 1)).float()
    x3 = (matches_fixed(x2, w3, 0) > t3.view(1, -1, 1, 1)).float()          # 每通道各自 θ
    s = F.avg_pool2d(x3, m.pool) * (m.pool ** 2)
    fh = (s >= m.pool ** 2 / 2.0).float().flatten(1)                       # 每通道多数表决
    mf = matches_fixed(fh.reshape(-1, m.feat, 1, 1), wf.reshape(4, m.feat, 1, 1), 0)
    score = 2.0 * mf.reshape(-1, 4) - m.feat                               # ±feat 的整数得分
    bias_q = torch.round(m.fc.bias.detach() * fc_scale).long()
    ex_int = score.long() * fc_scale + bias_q.view(1, 4)
    ex_logit = score + bias_q.float().view(1, 4) / fc_scale
    return ex_logit, ex_int


# ------------------------------------------------------------------ 评估
def evaluate(m, X, y, tag='', ep=0, strict=True, theta_quant='floor', fc_scale=SCALE_FC):
    m.eval()
    with torch.no_grad():
        fl = m(X)
        ex_logit, ex_int = hw_eval8(m, X, theta_quant, fc_scale)
        ref = T10.hw_eval(m, X)                     # 参考实现交叉校验（float bias）
    fa = fl.argmax(1)
    ea = ex_int.argmax(1)
    ia = ex_logit.argmax(1)
    ra = ref.argmax(1)
    n = int(len(y))
    out = dict(float=float((fa == y).float().mean()),
               exact=float((ea == y).float().mean()),
               agree=int((fa == ea).sum()),
               agree_int=int((ea == ia).sum()),
               agree_ref=int((fa == ra).sum()),
               n=n, maxdiff=float((fl - ex_logit).abs().max()))
    if strict:
        assert out['agree_int'] == n, '[%s ep%d] 整数/实值得分 argmax 不一致 %d/%d' % (
            tag, ep, n - out['agree_int'], n)
        assert out['agree'] == n, '[%s ep%d] float/exact argmax 不一致 %d/%d -- 本次评估不采信' % (
            tag, ep, n - out['agree'], n)
        assert out['agree_ref'] == n, (
            '[%s ep%d] 自实现 hw_eval 与 train_bnn10.hw_eval argmax 不一致 %d/%d'
            % (tag, ep, n - out['agree_ref'], n))
    return out


# ------------------------------------------------------------------ 初始化
def warm_start(m, teacher_path=TEACHER, fc_init='rand'):
    """teacher 初始化 + c3 4->cout3 通道扩展。

    忠实于 train_fix.run_one：c1/dw/c3 权重 + b1/b2/b3 全量复制；fc 默认不复制
    （train_fix.run_one 只复制卷积与 BN，fc 保持随机初始化）。fc_init='teacher' 时才
    按层平铺复制（train_honest 的口径），用于对照。
    c3 第 5~cout3 个通道 = teacher 通道 + 小幅扰动（否则完全对称、永不分化）。
    本函数不自行 seed：调用方已 torch.manual_seed(seed)，RNG 消耗顺序与 train_fix 一致。
    """
    tsd = torch.load(teacher_path, map_location='cpu')
    cnn = train_honest.BNNet()
    cnn.load_state_dict(tsd)
    ncopy = min(4, m.c3.weight.shape[0])
    with torch.no_grad():
        m.c1.weight.copy_(cnn.c1.weight)
        m.dw.weight.copy_(cnn.dw.weight)
        m.c3.weight[:ncopy].copy_(cnn.c3.weight[:ncopy])
        extra = m.c3.weight.shape[0] - ncopy
        if extra > 0:
            reps = (extra + 3) // 4
            base = cnn.c3.weight.repeat(reps, 1, 1, 1)[:extra]
            m.c3.weight[ncopy:].copy_(base + 0.10 * torch.randn_like(base))
        for k in ['weight', 'bias', 'running_mean', 'running_var']:
            getattr(m.b1, k).copy_(getattr(cnn.b1, k))
            getattr(m.b2, k).copy_(getattr(cnn.b2, k))
            v = getattr(cnn.b3, k)
            getattr(m.b3, k)[:ncopy].copy_(v[:ncopy])
            if extra > 0:
                vbase = v.repeat(reps)[:extra]
                getattr(m.b3, k)[ncopy:].copy_(vbase + 0.01 * torch.randn_like(vbase))
        if fc_init == 'teacher':
            half = cnn.fc.weight.shape[1]
            nch = m.fc.weight.shape[1] // half
            assert nch * half == m.fc.weight.shape[1], (half, m.fc.weight.shape)
            for j in range(nch):
                w = cnn.fc.weight / nch
                if j > 0:
                    w = w + 0.02 * torch.randn_like(w)
                m.fc.weight[:, j * half:(j + 1) * half].copy_(w)
            m.fc.bias.copy_(cnn.fc.bias)
    return m


# ------------------------------------------------------------------ 训练
def train(args, Xtr, ytr, Xte, yte):
    tag = args.tag or ('%s_c%d_%dep_s%d' % (args.init, args.cout3, args.epochs, args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    m = Net8(POOL, args.cout3)
    if args.init == 'teacher':
        warm_start(m, TEACHER, args.fc_init)
    nparam = sum(p.numel() for p in m.parameters())
    print('[model] tag=%s L3 out=%d feat=%d FC=4x%d params=%d pool=%d fc_init=%s'
          % (tag, args.cout3, m.feat, m.feat, nparam, POOL,
             args.fc_init if args.init == 'teacher' else 'random'), flush=True)

    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    best = dict(exact=-1.0, ep=-1, state=None)
    hist = []

    def do_eval(ep, loss=None, tr=None):
        r = evaluate(m, Xte, yte, tag, ep, strict=False)
        r['ep'] = ep
        r['valid'] = (r['agree'] == r['n'])
        hist.append(r)
        msg = '[%s ep%02d] ' % (tag, ep)
        if loss is not None:
            msg += 'loss=%.3f train=%.4f | ' % (loss, tr)
        msg += ('float=%.4f exact=%.4f agree=%d/%d agree_ref=%d/%d maxdiff=%.2e'
                % (r['float'], r['exact'], r['agree'], r['n'], r['agree_ref'], r['n'],
                   r['maxdiff']))
        if not r['valid']:
            msg += '  [INVALID agree %d/%d]' % (r['n'] - r['agree'], r['n'])
        elif r['exact'] > best['exact']:
            best.update(exact=r['exact'], ep=ep, state=copy.deepcopy(m.state_dict()))
            msg += '  <= best'
        print(msg, flush=True)
        return r

    do_eval(0)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        m.train()
        perm = torch.randperm(n)
        tot = 0.0
        corr = 0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            out = m(Xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            corr += (out.argmax(1) == ytr[idx]).sum().item()
        sch.step()
        if ep % args.eval_every == 0 or ep == args.epochs:
            do_eval(ep, tot / n, corr / n)
        if ep % 5 == 0:
            print('[%s] %d/%d ep, %.1f s elapsed (%.1f s/ep)'
                  % (tag, ep, args.epochs, time.time() - t0,
                     (time.time() - t0) / ep), flush=True)

    n_invalid = sum(0 if r['valid'] else 1 for r in hist)
    worst = min((r['agree'] for r in hist), default=0)
    print('[AGREE] evals=%d  invalid=%d  worst_agree=%d/%d'
          % (len(hist), n_invalid, worst, hist[0]['n']), flush=True)
    if best['state'] is None:
        print('[FAIL] 没有任何一次评估通过 float==exact 逐点一致门，本次结果不采信', flush=True)
        return -1.0, hist
    m.load_state_dict(best['state'])
    rv = evaluate(m, Xte, yte, tag, best['ep'], strict=True)      # 最终严格门
    print('[VERIFY] reloaded best ep%d: float=%.4f exact=%.4f agree=%d/%d -> PASS'
          % (best['ep'], rv['float'], rv['exact'], rv['agree'], rv['n']), flush=True)
    bi = next(r for r in hist if r['ep'] == best['ep'])
    print('[RESULT] tag=%s init=%s fc_init=%s epochs=%d lr=%g seed=%d best_exact=%.4f '
          'best_ep=%d best_float=%.4f agree=%d/%d'
          % (tag, args.init, args.fc_init, args.epochs, args.lr, args.seed, best['exact'],
             best['ep'], bi['float'], bi['agree'], bi['n']), flush=True)
    print('[HIST] ' + ' '.join('ep%d:%.4f' % (r['ep'], r['exact']) for r in hist), flush=True)
    if args.out:
        if best['exact'] >= args.out_min:
            torch.save(m.state_dict(), args.out)
            print('[save] %s (best ep%d exact=%.4f)' % (args.out, best['ep'], best['exact']),
                  flush=True)
        else:
            print('[skip-save] best_exact=%.4f < --out-min %.4f，保留已有文件 %s'
                  % (best['exact'], args.out_min, args.out), flush=True)
    return best['exact'], hist


# ------------------------------------------------------------------ 诊断
def _float_layers(m, X):
    x = F.pad(X, (1,) * 4, value=0.0)
    a1 = torch.sign(m.b1(F.conv2d(x, torch.sign(m.c1.weight))))
    x = F.pad(a1, (1,) * 4, value=-1.0)
    a2 = torch.sign(m.b2(F.conv2d(x, torch.sign(m.dw.weight), groups=8)))
    a3 = torch.sign(m.b3(F.conv2d(a2, torch.sign(m.c3.weight))))
    return a1, a2, a3


def _exact_layers(m, X):
    w1, w2, w3 = [(p > 0).float() for p in (m.c1.weight, m.dw.weight, m.c3.weight)]
    t1 = theta_of(m.b1, m.c1.weight, 9, False)
    t2 = theta_of(m.b2, m.dw.weight, 9, True)
    t3 = theta_of(m.b3, m.c3.weight, 8, True)
    e1 = (matches_fixed(X, w1, 1) > t1.view(1, -1, 1, 1)).float()
    e2 = (matches_fixed(e1, w2, 1, groups=8) > t2.view(1, -1, 1, 1)).float()
    e3 = (matches_fixed(e2, w3, 0) > t3.view(1, -1, 1, 1)).float()
    return (e1, e2, e3), (t1, t2, t3)


def diag(Xte, yte):
    """teacher 初始化点的逐层比对：区分"FC 偏置量化"与"激活位翻转"。"""
    torch.manual_seed(1)
    m = Net8(POOL, N_OUT)
    warm_start(m, TEACHER)
    m.eval()
    with torch.no_grad():
        fl = m(Xte)
        a = _float_layers(m, Xte)
        e, t = _exact_layers(m, Xte)
        ex_f = T10.hw_eval(m, Xte)
        ex_l, ex_i = hw_eval8(m, Xte)
    n = len(yte)
    fa, ra, ea = fl.argmax(1), ex_f.argmax(1), ex_i.argmax(1)
    print('[diag] n=%d  theta=%s' % (n, [np.round(x.numpy(), 3).tolist() for x in t]))
    print('[diag] theta floor 与 float theta 相同? %s'
          % all(torch.equal(x, torch.floor(x)) for x in t))
    print('[diag] float vs T10.hw_eval(float bias): %d/%d' % (int((fa == ra).sum()), n))
    print('[diag] float vs hw_eval8(Q8.8 bias)    : %d/%d' % (int((fa == ea).sum()), n))
    for name, aa, ee in zip(('L1', 'L2', 'L3'), a, e):
        d = ((aa > 0.5) ^ (ee > 0.5))
        print('[diag] %s pixel mismatch: %d / %d' % (name, int(d.sum()), d.numel()))
    print('[diag] sign(0) 病态激活数: %d' % int(sum((x == 0).sum() for x in a)))


def diag_rand(Xte, yte):
    """随机初始化（BN 还是 μ=0,σ=1,β=0,γ=1）时的 float/exact 分歧来源。"""
    torch.manual_seed(1)
    m = Net8(POOL, N_OUT)
    m.eval()
    with torch.no_grad():
        fl = m(Xte)
        a = _float_layers(m, Xte)
        e, t = _exact_layers(m, Xte)
        ex_l, ex_i = hw_eval8(m, Xte)
    n = len(yte)
    print('[diag-rand] n=%d  float-exact agree=%d/%d  maxdiff=%.2e'
          % (n, int((fl.argmax(1) == ex_i.argmax(1)).sum()), n,
             float((fl - ex_l).abs().max())))
    for name, aa, ee in zip(('L1', 'L2', 'L3'), a, e):
        d = ((aa > 0.5) ^ (ee > 0.5))
        print('[diag-rand] %s: sign(0) 像素=%d  像素级不一致=%d/%d'
              % (name, int((aa == 0).sum()), int(d.sum()), d.numel()))
    print('[diag-rand] theta=%s' % [np.round(x.numpy(), 3).tolist() for x in t])


def diag_theta(m, X):
    """同一层输入下，比较 float θ 与 floor(θ) 的逐层判决差异，定位 θ 量化的真实敏感性。"""
    w1, w2, w3 = [(p > 0).float() for p in (m.c1.weight, m.dw.weight, m.c3.weight)]
    t1 = theta_of(m.b1, m.c1.weight, 9, False)
    t2 = theta_of(m.b2, m.dw.weight, 9, True)
    t3 = theta_of(m.b3, m.c3.weight, 8, True)
    x = X
    for name, xx, w, t, pad, groups in [('L1', X, w1, t1, 1, 1),
                                       ('L2', None, w2, t2, 1, 8),
                                       ('L3', None, w3, t3, 0, 1)]:
        if name == 'L2':
            xx = (matches_fixed(X, w1, 1) > t1.view(1, -1, 1, 1)).float()
        if name == 'L3':
            xx = (matches_fixed(matches_fixed(X, w1, 1) > t1.view(1, -1, 1, 1)
                                , w2, 1, groups=8) > t2.view(1, -1, 1, 1)).float()
        mm = matches_fixed(xx, w, pad, groups)
        a = (mm > t.view(1, -1, 1, 1)).float()
        b = (mm > torch.floor(t).view(1, -1, 1, 1)).float()
        c = (mm >= torch.ceil(t).view(1, -1, 1, 1)).float()
        d = (a != b)
        e = (a != c)
        frac = float((mm - mm.round()).abs().max())
        ch = d.any(dim=(0, 2, 3)).nonzero().flatten().tolist()
        print('[diag-theta] %s | matches 最大小数偏差=%.3e | floor 不一致=%d | ceil 不一致=%d '
              '| theta=%s | floor=%s | ceil=%s'
              % (name, frac, int(d.sum()), int(e.sum()),
                 np.round(t.numpy(), 4).tolist(), np.round(torch.floor(t).numpy(), 2).tolist(),
                 np.round(torch.ceil(t).numpy(), 2).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', choices=['teacher', 'rand'], default='teacher')
    ap.add_argument('--fc-init', choices=['rand', 'teacher'], default='rand',
                    help='fc 初始化：rand=train_fix.run_one 口径（fc 随机），teacher=平铺复制')
    ap.add_argument('--cout3', type=int, default=N_OUT,
                    help='L3 输出通道数（4=复现基线，8=本实验）')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--out-min', type=float, default=0.0,
                    help='仅当 best_exact >= out_min 时才写 --out（防止用更差的模型覆盖）')
    ap.add_argument('--tag', type=str, default='')
    ap.add_argument('--diag', action='store_true', help='teacher 初始化点的逐层分歧诊断')
    ap.add_argument('--diag-rand', action='store_true', help='随机初始化点的 sign(0) 分歧诊断')
    ap.add_argument('--diag-theta', type=str, default='', help='对给定 checkpoint 做 θ/floor(θ) 逐层差异诊断')
    ap.add_argument('--verify', type=str, default='', help='只复验该 checkpoint')
    ap.add_argument('--theta-quant', choices=['none', 'floor', 'u8'], default='none',
                    help='θ 硬件存储口径（仅复验用）：floor=整数部分（二值输入下与 float 等价），'
                         'u8=8bit 步长 1/8（反例）')
    ap.add_argument('--fc-scale', type=int, default=SCALE_FC,
                    help='FC 整数偏置标度：256=Q8.8（默认），1=规范 v3 的 9bit 整数偏置')
    a = ap.parse_args()
    assert torch.get_num_threads() == 4, 'CPU 纪律：必须 4 线程'

    t0 = time.time()
    Xtr, ytr, Xte, yte = train_fix.build_split(seed=7, holdout=0.2)
    print('[data] 与 train_fix.build_split(seed=7, holdout=0.2) 完全相同')
    print('[data] train=%d test=%d  train_per_class=%s  test_per_class=%s'
          % (len(Xtr), len(Xte),
             np.bincount(ytr.numpy(), minlength=4).tolist(),
             np.bincount(yte.numpy(), minlength=4).tolist()), flush=True)
    assert len(Xtr) == 6432 and len(Xte) == 1104, (len(Xtr), len(Xte))
    # 输入是否为严格 0/1：junk 的 crop->resize 路径产生连续灰度，硬件 1bit FRAME 无法表示
    nb_tr = int(((Xtr != 0) & (Xtr != 1)).flatten(1).any(1).sum())
    nb_te = int(((Xte != 0) & (Xte != 1)).flatten(1).any(1).sum())
    print('[data] 非二值(含连续灰度)样本: train=%d/%d  test=%d/%d'
          % (nb_tr, len(Xtr), nb_te, len(Xte)), flush=True)
    import hashlib
    print('[data] sha1(Xtr)=%s sha1(Xte)=%s sha1(ytr)=%s sha1(yte)=%s (%.1fs)'
          % (hashlib.sha1(Xtr.numpy().tobytes()).hexdigest()[:16],
             hashlib.sha1(Xte.numpy().tobytes()).hexdigest()[:16],
             hashlib.sha1(ytr.numpy().tobytes()).hexdigest()[:16],
             hashlib.sha1(yte.numpy().tobytes()).hexdigest()[:16], time.time() - t0),
          flush=True)

    if a.diag:
        diag(Xte, yte)
        return
    if a.diag_rand:
        diag_rand(Xte, yte)
        return
    if a.diag_theta:
        m = Net8(POOL, a.cout3)
        m.load_state_dict(torch.load(a.diag_theta, map_location='cpu'))
        m.eval()
        with torch.no_grad():
            diag_theta(m, Xte)
        return
    if a.verify:
        m = Net8(POOL, a.cout3)
        m.load_state_dict(torch.load(a.verify, map_location='cpu'))
        r = evaluate(m, Xte, yte, 'verify', -1, strict=False, theta_quant=a.theta_quant,
                     fc_scale=a.fc_scale)
        print('[VERIFY-FILE] %s  L3out=%d feat=%d theta=%s fc_scale=%d | float=%.4f '
              'exact=%.4f agree=%d/%d maxdiff=%.2e'
              % (a.verify, a.cout3, m.feat, a.theta_quant, a.fc_scale, r['float'],
                 r['exact'], r['agree'], r['n'], r['maxdiff']), flush=True)
        if a.theta_quant == 'none':
            assert r['agree'] == r['n'], 'float/exact 不一致 %d/%d' % (r['n'] - r['agree'], r['n'])
            print('[VERIFY-FILE] 严格门 PASS（float==exact 逐点相同）', flush=True)
        return

    best, hist = train(a, Xtr, ytr, Xte, yte)
    print('[DONE] best_exact=%.4f  total=%.1fs' % (best, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
