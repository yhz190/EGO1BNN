#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
极简 PDF 文本/关键词抽取器（无第三方依赖，v2）
修复：① 兼容 1 字节简单字体（英文/数字不再丢）② 非法 hex 不再崩
      ③ 新增关键词模式：直接扫解压流里的 ASCII 串（找 Vivado 版本/芯片型号最好用）
用法：
  python pdf_text.py <pdf> text <out.txt>       # 抽正文
  python pdf_text.py <pdf> grep Vivado,FT,Type  # 按关键词找 ASCII 串
"""
import re, sys, zlib, os

def decompress_streams(data):
    out = []
    for m in re.finditer(rb'stream\r?\n', data):
        s = m.end()
        e = data.find(b'endstream', s)
        if e < 0:
            continue
        raw = data[s:e]
        for trim in (0, 1, 2, 4):
            try:
                out.append(zlib.decompress(raw[:len(raw) - trim] if trim else raw))
                break
            except Exception:
                continue
    return out

def parse_tounicode(streams):
    cmap = {}
    for s in streams:
        if b'beginbfchar' not in s and b'beginbfrange' not in s:
            continue
        for blk in re.findall(rb'beginbfchar(.*?)endbfchar', s, re.S):
            for a, b in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                try:
                    cmap[int(a, 16)] = bytes.fromhex(b.decode()).decode('utf-16-be', 'ignore')
                except Exception:
                    pass
        for blk in re.findall(rb'beginbfrange(.*?)endbfrange', s, re.S):
            for a, b, c in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                try:
                    lo, hi, base = int(a, 16), int(b, 16), int(c, 16)
                except Exception:
                    continue
                if hi < lo or hi - lo > 65535:
                    continue
                for i in range(hi - lo + 1):
                    cmap[lo + i] = chr(base + i)
    return cmap

def decode_any(h, cmap):
    """自动判断 1 字节还是 2 字节编码"""
    h = re.sub(rb'\s', b'', h).decode('ascii', 'ignore')
    if len(h) % 2:
        h = '0' + h
    try:
        raw = bytes.fromhex(h)
    except Exception:
        return ''
    if len(raw) == 0:
        return ''
    if len(raw) % 2 == 0:
        codes = [(raw[i] << 8) | raw[i + 1] for i in range(0, len(raw), 2)]
        hit = sum(1 for c in codes if c in cmap)
        if hit >= max(1, len(codes) // 2):
            return ''.join(cmap.get(c, '') for c in codes)
    # 退回 1 字节
    return ''.join(cmap.get(b, chr(b) if 32 <= b < 127 else '') for b in raw)

def extract_text(streams, cmap):
    pieces = []
    for s in streams:
        if b'BT' not in s:
            continue
        for blk in re.findall(rb'BT(.*?)ET', s, re.S):
            txt = []
            for m in re.finditer(rb'<([0-9A-Fa-f\s]+)>|(T[dD]|T\*|TD)', blk, re.S):
                if m.group(1):
                    txt.append(decode_any(m.group(1), cmap))
                elif m.group(2):
                    txt.append('\n')
            p = ''.join(txt)
            p = re.sub(r'\n{3,}', '\n\n', p)
            if p.strip():
                pieces.append(p)
    return '\n'.join(pieces)

def grep_ascii(streams, kws):
    hits = set()
    for s in streams:
        for m in re.finditer(rb'[ -~]{3,}', s):
            t = m.group(0).decode('ascii', 'ignore')
            if any(k.lower() in t.lower() for k in kws):
                hits.add(t.strip())
    return sorted(hits)

def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    path, mode = sys.argv[1], sys.argv[2]
    data = open(path, 'rb').read()
    streams = decompress_streams(data)
    cmap = parse_tounicode(streams)
    print("[info] %s | %.1fMB | 解压流 %d | CMap %d"
          % (os.path.basename(path), len(data) / 1048576, len(streams), len(cmap)))
    if mode == 'text':
        text = extract_text(streams, cmap)
        cn = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        print("[text] 字符 %d，其中中文 %d" % (len(text), cn))
        out = sys.argv[3] if len(sys.argv) > 3 else None
        if out:
            open(out, 'w', encoding='utf-8').write(text)
            print("[ok] ->", out)
        else:
            print(text[:2500])
    elif mode == 'actual':
        at = dump_actualtext(streams)
        print("[actual] 条目 %d" % len(at))
        seen = set()
        for t in at:
            if t in seen:
                continue
            seen.add(t)
            print(" ", t)
    else:
        kws = (sys.argv[3] if len(sys.argv) > 3 else "Vivado,Version,XC7,Artix,EGO1,FT2,CH3,Type-C,N25Q").split(',')
        for t in grep_ascii(streams, kws):
            print(" ", t[:160])

def dump_actualtext(streams):
    """带标记 PDF 里 /ActualText(...) 保存了原文（英文/数字都在），按顺序输出。
       注意：ActualText 在压缩的对象流里，必须扫解压后的流。"""
    out = []
    for data in streams:
        for m in re.finditer(rb'/ActualText\s*\(((?:\\.|[^\\()])*)\)', data):
            s = m.group(1)
            s = s.replace(b'\\(', b'(').replace(b'\\)', b')').replace(b'\\\\', b'\\')
            out.append(s.decode('utf-8', 'ignore'))
    return out

if __name__ == "__main__":
    main()
