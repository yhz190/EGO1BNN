
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""golden_hw.py — BNN 位精确黄金模型 + 逐层测试向量"""
import argparse
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
import train_bnn10 as T

torch.set_num_threads(1)

IMG, FB, NCLS, EXPECT = 64, 512, 4, 595


def load_vectors(d, order="msb"):
    with open(os.path.join(d, "labels.csv"), encoding="utf-8-sig") as fh:
        rows = [(r["file"].strip(), int(r["class"])) for r in csv.DictReader(fh)]
    if not rows:
        raise SystemExit("labels.csv 是空的")
    X = np.zeros((len(rows), 1, IMG, IMG), np.float32)
    for i, (fn, _) in enumerate(rows):
        raw = np.fromfile(os.path.join(d, fn), np.uint8)
        if raw.size != FB:
            raise SystemExit("%s 是 %d 字节，应为 %d" % (fn, raw.size, FB))
        b = np.unpackbits(raw, bitorder="little" if order == "lsb" else "big")
        X[i, 0] = b[: IMG * IMG].reshape(IMG, IMG)
    y = np.array([c for _, c in rows], np.int64)
    return rows, torch.from_numpy(X), torch.from_numpy(y)


@torch.no_grad()
def trace(m, X):
    w1 = (m.c1.weight > 0).float()
    w2 = (m.dw.weight > 0).float()
    w3 = (m.c3.weight > 0).float()
    wf = (m.fc.weight > 0).float()
    t1 = T.theta_of(m.b1, m.c1.weight, 9, False)
    t2 = T.theta_of(m.b2, m.dw.weight, 9, True)
    t3 = T.theta_of(m.b3, m.c3.weight, 8, True)
    m1 = T.matches_fixed(X, w1, 1)
    x1 = (m1 > t1.view(1, -1, 1, 1)).float()
    m2 = T.matches_fixed(x1, w2, 1, groups=8)
    x2 = (m2 > t2.view(1, -1, 1, 1)).float()
    m3 = T.matches_fixed(x2, w3, 0)
    x3 = (m3 > t3.view(1, -1, 1, 1)).float()
    s = F.avg_pool2d(x3, m.pool) * (m.pool ** 2)
    fh = (s >= m.pool ** 2 / 2.0).float().flatten(1)
    mf = T.matches_fixed(fh.reshape(-1, m.feat, 1, 1), wf.reshape(NCLS, m.feat, 1, 1), 0)
    lg = (2.0 * mf.reshape(-1, NCLS) - m.feat) + m.fc.bias
    return dict(m1=m1, x1=x1, x2=x2, x3=x3, fh=fh, logits=lg)


def hx(plane):
    out = []
    for c in plane:
        for row in c:
            v = 0
            for b in row:
                v = (v << 1) | int(b)
            out.append("%016x" % v)
    return out


def hx1(bits):
    bs = [int(b) for b in bits]
    out = []
    for i in range(0, len(bs), 64):
        c = bs[i:i + 64]
        c += [0] * (64 - len(c))
        v = 0
        for b in c:
            v = (v << 1) | b
        out.append("%016x" % v)
    return out


def wr(p, lines):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def dump(out, i, name, tr, ytrue, lg):
    d = os.path.join(out, "frame_%03d_%s" % (i, os.path.splitext(name)[0]))
    os.makedirs(d, exist_ok=True)

    cnt = tr["m1"][i].numpy()
    wr(os.path.join(d, "L1_cnt.txt"),
       [" ".join(str(int(v)) for v in cnt[c][y])
        for c in range(cnt.shape[0]) for y in range(cnt.shape[1])])

    wr(os.path.join(d, "L1_act.hex"), hx(tr["x1"][i].numpy()))
    wr(os.path.join(d, "L2_act.hex"), hx(tr["x2"][i].numpy()))
    wr(os.path.join(d, "L3_act.hex"), hx(tr["x3"][i].numpy()))
    wr(os.path.join(d, "L4_feat.hex"), hx1(tr["fh"][i].numpy()))

    sc = lg[i].tolist()
    am = int(lg[i].argmax())
    wr(os.path.join(d, "L5_score.txt"),
       ["s%d = %.6f" % (k, v) for k, v in enumerate(sc)]
       + ["argmax = %d   (true = %d)   %s"
          % (am, int(ytrue[i]), "OK" if am == int(ytrue[i]) else "WRONG")])


README_LINES = [
    "# golden 逐层向量格式说明",
    "",
    "由 src/pc/golden_hw.py 生成，供 RTL testbench 与 T8 逐层比对。",
    "",
    "## 目录",
    "- frame_XXX_<源文件名>/ : 每帧一个目录",
    "- summary.csv : 全部帧逐帧结果",
    "- report.json : 自检结论",
    "",
    "## 每帧文件",
    "- L1_cnt.txt  : L1 匹配计数，8 通道 x 64 行，每行 64 个十进制数(0~9)",
    "- L1_act.hex  : L1 激活位图，8x64 行，每行 16 个 hex(=64bit，MSB 在左)",
    "- L2_act.hex  : 同上，8x64 行",
    "- L3_act.hex  : 同上，4x64 行",
    "- L4_feat.hex : L4 池化后 1024bit，16 行 x 16 hex",
    "- L5_score.txt: 4 行得分 + 1 行 argmax",
    "",
    "## 约定",
    "- 行序 [channel][y]，x 从左到右，MSB 在左",
    "- 激活 bit=1 表示 +1，bit=0 表示 -1",
    "- 类别: 0=rock 1=paper 2=scissors 3=junk",
    "- hw_eval 的 FC 偏置是浮点，RTL 是整数 thetaFC：",
    "  L5 只比 argmax，L1~L4 位图必须 0 差异",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--all-frames", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--bit-order", choices=["msb", "lsb"], default="msb")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    vd = os.path.join(root, "sim", "vectors_t1a")
    ck = os.path.join(root, "src", "weights", "ckpt_ft_pool4_ep25.pt")
    out = os.path.join(root, "sim", "golden")
    for p in (vd, ck):
        if not os.path.exists(p):
            raise SystemExit("找不到 %s" % p)

    m = T.Net(4)
    try:
        sd = torch.load(ck, map_location="cpu")
    except Exception:
        sd = torch.load(ck, map_location="cpu", weights_only=False)
    m.load_state_dict(sd)
    m.eval()

    if a.probe:
        for o in ("msb", "lsb"):
            _, X, y = load_vectors(vd, o)
            with torch.no_grad():
                lg = T.hw_eval(m, X)
            k = int((lg.argmax(1) == y).sum())
            print("位序 %s : %d/%d = %.4f" % (o, k, len(y), k / len(y)))
        return 0

    rows, X, y = load_vectors(vd, a.bit_order)
    n = len(rows)
    print("[load] %d 帧  位序 %s  权重 %s" % (n, a.bit_order, os.path.basename(ck)))

    correct = 0
    hw_ok = True
    logs = []
    keep = []
    dump_idx = set(range(n)) if a.all_frames else set(range(min(a.frames, n)))

    for s in range(0, n, a.batch):
        xb = X[s:s + a.batch]
        tr = trace(m, xb)
        with torch.no_grad():
            hw = T.hw_eval(m, xb)
        if not torch.equal(tr["logits"], hw):
            hw_ok = False
            print("[!] 第 %d 批与 hw_eval 不一致（max diff %.3e）"
                  % (s, (tr["logits"] - hw).abs().max().item()))
        correct += int((tr["logits"].argmax(1) == y[s:s + a.batch]).sum())
        logs.append(tr["logits"].clone())
        keep.append(tr["logits"].clone())
        for i in range(xb.shape[0]):
            if s + i in dump_idx:
                dump(out, s + i, rows[s + i][0], tr, y[s:s + a.batch], tr["logits"])
        print("  %d/%d" % (min(s + a.batch, n), n), end="\r")
    print()

    logits = torch.cat(logs, 0)

    det = True
    for s in range(0, n, a.batch):
        if not torch.equal(trace(m, X[s:s + a.batch])["logits"], keep[s // a.batch]):
            det = False
            break

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "file", "true", "pred", "correct", "s0", "s1", "s2", "s3"])
        for i, (fn, cls) in enumerate(rows):
            pr = int(logits[i].argmax())
            w.writerow([i, fn, cls, pr, int(pr == cls)]
                       + ["%.6f" % v for v in logits[i].tolist()])

    wr(os.path.join(out, "README_vectors.md"), README_LINES)

    rep = {"total": n, "correct": correct, "accuracy": round(correct / n, 6),
           "expect_accuracy": 0.9597, "matches_hw_eval_bitwise": hw_ok,
           "deterministic": det, "bit_order": a.bit_order,
           "ckpt": os.path.basename(ck),
           "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "dumped_frames": len(dump_idx)}
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)

    print()
    print("准确率    : %d/%d = %.4f   (期望 595/620 = 0.9597)" % (correct, n, correct / n))
    print("与hw_eval : %s" % ("逐位一致" if hw_ok else "不一致"))
    print("确定性    : %s" % ("两次运行相同" if det else "两次运行不同"))
    print("输出目录  : %s" % out)

    if correct != EXPECT or not hw_ok or not det:
        print("\n[FAIL] 不满足验收。按规范第 6 节：停下报告差异，不要改数字凑。")
        return 1

    print("[PASS] 全部验收项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

