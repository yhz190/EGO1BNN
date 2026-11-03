#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_dsp.py -- dual gate for DSP48E1 == 0 and timing.

Gate 1 (post-synth):  DSP48E1 == 0 in synth_util.rpt
Gate 2 (post-route):  DSP48E1 == 0 in impl_util.rpt AND WNS >= 0 in impl_timing.rpt

usage: python check_dsp.py [build dir]
exit 0 = all gates pass
"""
import os
import re, sys, os, glob
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def dsp_count(path):
    if not os.path.exists(path):
        return None
    txt = open(path, encoding='utf-8', errors='ignore').read()
    # utilization rows: '| DSP48E1   |   0 |  0 | ...'
    m = re.search(r'\|\s*DSPs?\s*\|\s*(\d+)\s*\|', txt)
    if m:
        return int(m.group(1))
    m = re.search(r'DSP48E1\s*\|\s*(\d+)', txt)
    return int(m.group(1)) if m else None

def wns(path):
    if not os.path.exists(path):
        return None
    txt = open(path, encoding='utf-8', errors='ignore').read()
    # engine domain first, then general WNS column
    m = re.search(r'^\s*clk50\s+(-?\d+\.\d+)', txt, re.M)
    if m:
        return float(m.group(1))
    m = re.search(r'WNS\(ns\)\s*\n\s*-+\s*\n\s*(-?\d+\.\d+)', txt)
    if m:
        return float(m.group(1))
    m = re.search(r'Slack\s*\(MET\)\s*:\s*(-?\d+\.\d+)', txt)
    return float(m.group(1)) if m else None

def main():
    rep = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'build')
    checks = []
    s = dsp_count(os.path.join(rep, 'synth_util.rpt'))
    i = dsp_count(os.path.join(rep, 'impl_util.rpt'))
    w = wns(os.path.join(rep, 'impl_timing.rpt'))
    print('[gate1] post-synth  DSP48E1 =', s)
    print('[gate2] post-route  DSP48E1 =', i, ' WNS =', w)
    ok = True
    if s is None: print('[gate1] SKIP (report missing)'); ok = False
    elif s == 0:  print('[gate1] PASS')
    else:         print('[gate1] FAIL: DSP48E1 must be 0'); ok = False
    if i is None or w is None: print('[gate2] SKIP (report missing)'); ok = False
    elif i == 0 and w >= 0:    print('[gate2] PASS')
    else:                      print('[gate2] FAIL: DSP=%s WNS=%s' % (i, w)); ok = False
    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())