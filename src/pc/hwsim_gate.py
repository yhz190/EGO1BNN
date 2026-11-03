#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""hwsim_gate.py -- DECISIVE EXPERIMENT: exact hardware forward, global vs per-channel theta.

Hardware semantics (plan v2.2 S6.1):
  L1 3x3 1->8, XNOR popcount over 9 taps  -> compare theta -> 1bit
  L2 3x3 depthwise 8->8, popcount 9       -> theta
  L3 1x1 8->4, popcount 8                 -> theta
  L4 8x8 majority pool, fixed 32
  L5 FC 256->4, popcount                  -> argmax (+bias)

theta per channel derived EXACTLY from trained BN statistics:
  real_conv ~= alpha_c * (2*x - K),  x = match count (0..K), alpha_c = mean|W_c|
  BN(x) > 0  =>  gamma*(alpha*(2x-K) - mu)/sigma + beta > 0
             =>  x > (K + mu/alpha - beta*sigma/(gamma*alpha))/2 = theta_c

Variants:
  global  : one theta per layer (median of per-channel)
  perchan : theta_c per channel
"""
import os, sys, numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bnn2 import CNN, build_dataset

MODEL = os.path.join(ROOT, r'board/weights/model.pt')
DATA  = os.path.join(ROOT, r'data')

def matches(x_bin, w_bin, K, groups=1, pad=1):
    """integer XNOR-match count: conv(x,w) + conv(1-x,1-w)"""
    m1 = F.conv2d(x_bin, w_bin, None, 1, pad, groups=groups)
    m2 = F.conv2d(1 - x_bin, 1 - w_bin, None, 1, pad, groups=groups)
    return m1 + m2

def fold_theta(bn, w, K, sigma_scale=1.0):
    """theta in XNOR-match-count units for a model trained with 0/1 inputs.

    derivation (0/1 semantics):
      s_i = sign(w_i) in {+1,-1};  b_i in {0,1};  a_i = 2*b_i-1
      x (match count) = #{b_i == w_i} = (K + sum s_i*a_i)/2
      real_conv ~= alpha * sum s_i*b_i = alpha * (x - K/2 + sum(s_i)/2)
      BN>0  =>  real_conv > mu - beta*sigma/gamma
      =>  x > K/2 - sum(s_i)/2 + (mu - beta*sigma/gamma)/alpha
    """
    gamma = bn.weight.detach().clamp(min=1e-6)
    beta = bn.bias.detach()
    mu = bn.running_mean.detach()
    sigma = (bn.running_var.detach() + bn.eps).sqrt()
    if w.dim() == 4:
        alpha = w.detach().abs().mean(dim=(1, 2, 3))
        s = w.detach().sign()
        sum_s = s.sum(dim=(1, 2, 3))
    else:
        alpha = w.detach().abs().mean(dim=1)
        s = w.detach().sign()
        sum_s = s.sum(dim=1)
    alpha = alpha.clamp(min=1e-6)
    th = K / 2.0 - sum_s / 2.0 + (mu - beta * sigma / gamma) / alpha
    return th

def hw_forward(m, X_bin, thetas, mode='perchan'):
    """X_bin: (N,1,64,64) in {0,1}; returns logits (N,4)"""
    w1 = (m.c1.weight.detach() > 0).float(); K1 = 9
    w2 = (m.dw.weight.detach() > 0).float(); K2 = 9
    w3 = (m.c3.weight.detach() > 0).float(); K3 = 8
    wf = (m.fc.weight.detach() > 0).float()

    t1, t2, t3 = thetas['t1'], thetas['t2'], thetas['t3']
    if mode == 'global':
        t1 = t1.median().expand_as(t1); t2 = t2.median().expand_as(t2); t3 = t3.median().expand_as(t3)

    x1 = (matches(X_bin, w1, K1) > t1.view(1, -1, 1, 1)).float()
    x2 = (matches(x1, w2, K2, groups=8) > t2.view(1, -1, 1, 1)).float()
    x3 = (matches(x2, w3, K3, pad=0) > t3.view(1, -1, 1, 1)).float()
    # L4: 8x8 majority, fixed 32
    p = F.avg_pool2d(x3, 8) * 64.0
    feat = (p >= 32.0).float().flatten(1)
    # L5: match count -> logit
    mf = matches(feat.view(-1, 256, 1, 1), wf.view(4, 256, 1, 1), 256, pad=0).view(-1, 4)
    alpha_fc = (m.fc.weight.detach().abs().mean(dim=1)).clamp(min=1e-6)
    logit = alpha_fc.view(1, -1) * (2 * mf - 256) + m.fc.bias.detach().view(1, -1)
    return logit

def main():
    m = CNN(); m.load_state_dict(torch.load(MODEL, map_location='cpu')); m.eval()
    roots = [os.path.join(DATA, s) for s in ['rps', 'rps-test-set'] if os.path.isdir(os.path.join(DATA, s))]
    Xtr, ytr = build_dataset(roots, 124, seed=99, augment=False)   # test set (620)
    print('[data] test=%d  per-class=%s' % (len(Xtr), np.bincount(ytr.numpy(), minlength=4).tolist()))

    thetas = {
        't1': fold_theta(m.b1, m.c1.weight, 9),
        't2': fold_theta(m.b2, m.dw.weight, 9),
        't3': fold_theta(m.b3, m.c3.weight, 8),
    }
    for k, v in thetas.items():
        print('[theta] %s per-channel range [%.2f .. %.2f]  median=%.2f' % (k, float(v.min()), float(v.max()), float(v.median())))

    for mode in ['global', 'perchan']:
        with torch.no_grad():
            out = hw_forward(m, Xtr, thetas, mode)
        acc = (out.argmax(1) == ytr).float().mean().item()
        print('[RESULT] %-8s theta  binary-exact test accuracy = %.4f' % (mode, acc))

if __name__ == '__main__':
    main()
def search_theta(m, X, y=None, rounds=2):
    """coordinate search of integer per-channel theta on data, maximizing final accuracy"""
    thetas = {
        't1': fold_theta(m.b1, m.c1.weight, 9).round(),
        't2': fold_theta(m.b2, m.dw.weight, 9).round(),
        't3': fold_theta(m.b3, m.c3.weight, 8).round(),
    }
    best = evaluate(m, X, y, thetas, 'perchan')
    for r in range(rounds):
        for k, K in [('t1', 9), ('t2', 9), ('t3', 8)]:
            v = thetas[k].clone()
            for c in range(v.numel()):
                cur = v[c].item()
                for cand in range(0, K + 2):
                    v[c] = float(cand)
                    acc = evaluate(m, X, y, dict(thetas, **{k: v}), 'perchan')
                    if acc > best:
                        best = acc
                    else:
                        v[c] = cur if acc < best else v[c]
            thetas[k] = v
        print('  [search r%d] best=%.4f' % (r + 1, best))
    return thetas, best

def evaluate(m, X, y, thetas, mode):
    with torch.no_grad():
        out = hw_forward(m, X, thetas, mode)
    return (out.argmax(1) == y).float().mean().item()