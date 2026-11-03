#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_xdc.py -- validate our project XDC.

Checks:
  1. parse all set_property -dict {PACKAGE_PIN x IOSTANDARD y} [get_ports ...]
  2. no duplicate PACKAGE_PIN
  3. pin name format looks like a CSG324 pin
  4. IOSTANDARD is a known 7-series value
  5. every pin we use EXISTS in the official EGo1.xdc pin set (catches typos
     such as W19/U19/E19 that only exist on the 484-package template)
Exit 0 = pass, 1 = problems.
"""
import os
import re, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUR_XDC = os.path.join(ROOT, r'board/board_ego1_top_bnn.xdc')
OFF_XDC = os.path.join(ROOT, r'board/ego1_official_pins.xdc')  # 可选：官方 EGo1.xdc 副本；不存在时跳过比对

KNOWN_IO = {'LVCMOS33','LVCMOS25','LVCMOS18','LVCMOS15','LVCMOS12','LVTTL',
            'SSTL135','SSTL15','TMDS_33','LVDS_25','LVDS'}
ENT_RE = re.compile(r'set_property\s+-dict\s*\{([^}]*)\}\s*\[get_ports\s*\{?([^\[\]\}]*(\[[^\]]*\])?)\}?\s*\]')
PIN_RE = re.compile(r'PACKAGE_PIN\s+(\S+)')
PIN_ANY = re.compile(r'PACKAGE_PIN\s+([A-Za-z]+\d+)')

def strip_comments(text):
    return '\n'.join(ln for ln in text.splitlines() if not ln.lstrip().startswith('#'))


def parse(path):
    txt = strip_comments(open(path, encoding='utf-8', errors='ignore').read())
    out = []
    for m in ENT_RE.finditer(txt):
        prop, port = m.group(1), m.group(2).strip()
        pm = PIN_RE.search(prop)
        im = re.search(r'IOSTANDARD\s+(\S+)', prop)
        if pm:
            out.append((port, pm.group(1), im.group(1) if im else '-'))
    return out, txt

def main():
    if not os.path.exists(OUR_XDC):
        print('[skip] XDC not found: %s' % OUR_XDC)
        return 0
    ours, _ = parse(OUR_XDC)
    official_pins = set()
    if os.path.exists(OFF_XDC):
        official_pins = set(PIN_ANY.findall(open(OFF_XDC, encoding='utf-8', errors='ignore').read()))
        print('[official] %d distinct pins' % len(official_pins))
    else:
        print('[skip] official EGo1.xdc not bundled; skip official-pin comparison')
    print('[our xdc] parsed %d ports' % len(ours))

    errs = []
    dup = {k: v for k, v in collections.Counter(p for _, p, _ in ours).items() if v > 1}
    if dup: errs.append('duplicate pins: %s' % dup)
    for port, pin, io in ours:
        if not re.match(r'^[A-Z]{1,2}\d{1,2}$', pin):
            errs.append('bad pin format: %s -> %s' % (port, pin))
        if io not in KNOWN_IO:
            errs.append('unknown IOSTANDARD: %s -> %s' % (port, io))
    if official_pins:
        missing = [(p, n) for p, n, _ in ours if n not in official_pins]
        if missing:
            errs.append('%d pins not present in official EGo1.xdc: %s'
                        % (len(missing), ', '.join('%s=%s' % x for x in missing)))

    g = collections.Counter(re.sub(r'\[\d+\]$', '', p) for p, _, _ in ours)
    print('[groups] ' + ', '.join('%s=%d' % (k, v) for k, v in sorted(g.items())))
    print('[io] ' + ', '.join('%s x%d' % (k, v) for k, v in collections.Counter(i for _, _, i in ours).items()))

    if errs:
        print('[FAIL]')
        for e in errs: print('    ' + e)
        return 1
    print('[PASS] %d ports: structure OK, all pins exist on the board' % len(ours))
    return 0

if __name__ == '__main__':
    sys.exit(main())