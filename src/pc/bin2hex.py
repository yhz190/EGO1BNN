#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bin2hex.py —— 把固件二进制转成 32 位小端 hex（供 Vivado $readmemh 初始化 IMEM）"""
import sys, os, struct

def main():
    if len(sys.argv) < 3:
        print("usage: bin2hex.py <in.bin> <out.hex>"); return 1
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, 'rb').read()
    if len(data) % 4:
        data += b'\x00' * (4 - len(data) % 4)
    words = struct.unpack('<%dI' % (len(data)//4), data)
    with open(dst, 'w', encoding='ascii') as f:
        for w in words:
            f.write('%08x\n' % w)
    print("[ok] %s -> %s : %d bytes = %d words (%d fetch cycles)"
          % (src, dst, len(data), len(words), len(words)))
    return 0

if __name__ == '__main__':
    sys.exit(main())