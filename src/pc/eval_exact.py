#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_exact.py -- correct exact-integer hardware evaluation for sign-weight nets.

theta (popcount units) for a sign-weight net with BN before sign():
    conv_out = sum s_i*b_i   (s_i = sign(w_i) = +/-1, b_i in {0,1})
    matches m  = #(b_i == w_i) ;  sum s_i*a_i = 2m - K  (a_i = 2b_i-1)
    conv_out = (2m - K + sum s_i)/2
    sign(BN(conv_out)) = +1  <=>  conv_out > mu - beta*sigma/gamma
    => m > K/2 - sum(s_i)/2 + (mu - beta*sigma/gamma)
    NOTE: no division by alpha (alpha only applies to real-valued weights).
"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bnn2 import build_dataset
from hwsim_gate import matches
from train_bnn5 import BNNet

def theta_sign(bn, w, K):
    gamma = bn.weight.detach().clamp(min=1e-6)
    beta = bn.bias.detach()
    mu = bn.running_mean.detach()
    sigma = (bn.running_var.detach() + bn.eps).sqrt()
    s = w.detach().sign()
    sum_s = s.sum(dim=(1, 2, 3)) if w.dim() == 4 else s.sum(dim=1)
    return K / 2.0 - sum_s / 2.0 + (mu - beta * sigma / gamma)

def hw_forward_sign(m, X, thetas, mode='perchan'):
    w1 = (m.c1.weight.detach() > 0).float(); K1 = 9
    w2 = (m.dw.weight.detach() > 0).float(); K2 = 9
    w3 = (m.c3.weight.detach() > 0).float(); K3 = 8
    wf = (m.fc.weight.detach() > 0).float()
    t1, t2, t3 = thetas
    if mode == 'global':
        t1 = t1.median().expand_as(t1); t2 = t2.median().expand_as(t2); t3 = t3.median().expand_as(t3)
    x1 = (matches(X, w1, K1) > t1.view(1, -1, 1, 1)).float()
    x2 = (matches(x1, w2, K2, groups=8) > t2.view(1, -1, 1, 1)).float()
    x3 = (matches(x2, w3, K3, pad=0) > t3.view(1, -1, 1, 1)).float()
    feat = (F.avg_pool2d(x3, 8) * 64.0 >= 32.0).float().flatten(1)
    mf = matches(feat.view(-1, 256, 1, 1), wf.view(4, 256, 1, 1), 256, pad=0).view(-1, 4)
    alpha_fc = m.fc.weight.detach().abs().mean(dim=1).clamp(min=1e-6)
    # fc weight sign + real bias; scale by mean|w|
    logit = alpha_fc.view(1, -1) * (2 * mf - 256) + m.fc.bias.detach().view(1, -1)
    return logit

def main():
    DATA = os.path.join(ROOT, r'data')
    roots = [os.path.join(DATA, s) for s in ['rps', 'rps-test-set'] if os.path.isdir(os.path.join(DATA, s))]
    Xte, yte = build_dataset(roots, 124, seed=99, augment=False)
    m = BNNet(); m.load_state_dict(torch.load(os.path.join(ROOT, r'board/weights/bnn5.pt'), map_location='cpu')); m.eval()
    thetas = (theta_sign(m.b1, m.c1.weight, 9), theta_sign(m.b2, m.dw.weight, 9), theta_sign(m.b3, m.c3.weight, 8))
    for i, t in enumerate(thetas):
        print('[theta L%d] range [%.2f .. %.2f] median=%.2f' % (i+1, float(t.min()), float(t.max()), float(t.median())))
    with torch.no_grad():
        for mode in ['perchan', 'global']:
            out = hw_forward_sign(m, Xte, thetas, mode)
            acc = (out.argmax(1) == yte).float().mean().item()
            print('[EXACT-HW] %-8s = %.4f' % (mode, acc))
        # also training-style forward (float BN) for reference
        out2 = m(Xte); print('[float-NN ] = %.4f' % (out2.argmax(1) == yte).float().mean().item())

if __name__ == '__main__':
    main()