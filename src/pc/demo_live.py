#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""demo_live.py -- real-time demo pipeline: camera -> 512B binary frame -> board -> result.

Display uses **tkinter** (the OpenCV build in this venv is headless and cannot imshow;
the capture tool uses tkinter for the same reason).

Modes:
  default      : 1280x720 diagnostic (class, scores, thr, bit1, ok/bad, fps, binary thumb)
  --rec        : 1920x1080 clean recording canvas: full-bleed camera + huge class label +
                 four score bars + fps badge   (use this when recording the demo video)
  --fullscreen : start fullscreen (also toggled with 'f' inside the window)

Keys: q / ESC = quit, f = toggle fullscreen
Usage:
  ..\\.venv\\Scripts\\python.exe demo_live.py COM6 921600 --rec --fullscreen
"""
import os, sys, time
import numpy as np
import serial
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
import preprocess as P1
from class_map import CLASS_MAP

ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
PORT = ARGS[0] if len(ARGS) > 0 else 'COM6'
BAUD = int(ARGS[1]) if len(ARGS) > 1 else 921600
REC = '--rec' in sys.argv
FULL = '--fullscreen' in sys.argv
GUIDE = REC and ('--no-guide' not in sys.argv)  # 录制模式默认显示目标框
HYSTN = 3 if REC else 1                      # 录制模式默认 3 帧滞回
for i, a in enumerate(sys.argv):
    if a == '--hyst' and i + 1 < len(sys.argv):
        HYSTN = max(1, int(sys.argv[i + 1]))
    elif a.startswith('--hyst='):
        HYSTN = max(1, int(a.split('=', 1)[1]))
WIN = 'EGO1 BNN live demo'
COL = {'rock': (90, 230, 90), 'paper': (90, 200, 255), 'scissors': (255, 170, 70), 'junk': (180, 180, 180)}
_FONT_DIR = os.path.join(os.environ.get('WINDIR', ''), 'Fonts')
FONT_CAND = [os.path.join(_FONT_DIR, 'msyhbd.ttc'), os.path.join(_FONT_DIR, 'msyh.ttc'),
             os.path.join(_FONT_DIR, 'simhei.ttf')]


def font(sz):
    for p in FONT_CAND:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else ((crc << 1) & 0xFF)
    return crc


def downlink(payload):
    body = bytes([0xA5, 0x10, 0x00, 0x02]) + payload
    return body + bytes([crc8(body)])


class Hyst:
    """3-frame hysteresis: switch the DISPLAYED class only after N consecutive frames agree.

    Motivation (measured on a real 355-frame sequence, 2026-11-01):
      per-frame display: accuracy 93.80%, adjacent-frame class flip rate 6.2%
      3-frame hysteresis: accuracy 94.65%, flip rate 2.5%
    The raw board result is never modified (logging/diagnosis still sees every frame).
    """

    def __init__(self, n=3):
        self.n = max(1, int(n))
        self.cls = None
        self.cnt = 0

    def update(self, cls, valid=True):
        if not valid or self.n <= 1:
            if valid:
                self.cls = cls
            return self.cls if self.cls is not None else cls
        if self.cls is None:
            self.cls = cls
            self.cnt = 0
        elif cls == self.cls:
            self.cnt = 0
        else:
            self.cnt += 1
            if self.cnt >= self.n:
                self.cls = cls
                self.cnt = 0
        return self.cls


class Demo:
    def __init__(self, root, cap, ser):
        self.root = root
        self.cap = cap
        self.ser = ser
        self.fps_hist = []
        self.n_ok = self.n_bad = 0
        self.full = FULL
        self.guide = GUIDE
        self.hyst = Hyst(HYSTN)
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        self.scale = min(1.0, (sw - 40) / 1920.0, (sh - 120) / 1080.0)
        self.lbl = tk.Label(root, bg='black')
        self.lbl.pack()
        self._update_title(root)
        root.bind('<Key>', self.on_key)
        if self.full:
            root.attributes('-fullscreen', True)
        self.tick()

    def _update_title(self, root=None):
        root = root or self.root
        root.title(WIN + '  |  q=quit  f=fullscreen  g=目标框:%s  h=滞回:%s'
                   % ('开' if self.guide else '关', '%d帧' % self.hyst.n if self.hyst.n > 1 else '关'))

    def on_key(self, ev):
        k = ev.keysym.lower()
        if k == 'q' or k == 'escape':
            self.root.destroy()
        elif k == 'f':
            self.full = not self.full
            self.root.attributes('-fullscreen', self.full)
        elif k == 'g':
            self.guide = not self.guide
            self._update_title()
        elif k == 'h':
            self.hyst = Hyst(1 if self.hyst.n > 1 else max(3, HYSTN))
            self._update_title()

    # ---------- one frame ----------
    def tick(self):
        ok, bgr = self.cap.read()
        if not ok:
            self.root.after(20, self.tick)
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        packed, bits, thr = P1.to_frame(Image.fromarray(rgb))
        t0 = time.time()
        self.ser.reset_input_buffer()
        self.ser.write(downlink(packed))
        rsp = self.ser.read(10)
        dt = time.time() - t0

        cls = 3
        scores = [0, 0, 0]
        flags = 0
        good = False
        if len(rsp) == 10 and rsp[0] == 0xA5 and rsp[1] == 0x83 and crc8(rsp[:9]) == rsp[9]:
            cls = int(rsp[2])
            scores = [int(b - 256 if b >= 128 else b) for b in rsp[3:6]]
            flags = int(rsp[6])
            good = True
            self.n_ok += 1
        else:
            self.n_bad += 1
        disp_cls = self.hyst.update(cls, valid=good) if (good or self.hyst.cls is not None) else cls
        self.fps_hist.append(1.0 / max(dt, 1e-3))
        if len(self.fps_hist) > 30:
            self.fps_hist.pop(0)
        fps = float(np.mean(self.fps_hist))

        if REC:
            pil = self.render_rec(rgb, disp_cls, scores, fps, good)
            if getattr(self, 'scale', 1.0) < 1.0:
                pil = pil.resize((int(pil.width * self.scale), int(pil.height * self.scale)),
                                 Image.LANCZOS)
            w, h = pil.size
            if not self.full:
                self.root.geometry('%dx%d+0+0' % (w, h))
        else:
            pil = self.render_diag(rgb, disp_cls, scores, fps, thr, bits, good, flags)
        self.tk = ImageTk.PhotoImage(pil)
        self.lbl.config(image=self.tk)
        self.root.after(1, self.tick)

    # ---------- recording canvas ----------
    def render_rec(self, rgb, cls, scores, fps, good):
        W, H = 1920, 1080
        pil = Image.fromarray(rgb).resize((W, H), Image.BILINEAR)
        d = ImageDraw.Draw(pil)
        name = CLASS_MAP[cls] if 0 <= cls < 4 else '?'
        color = COL.get(name, (255, 255, 255)) if good else (255, 70, 70)
        # top translucent band (crop -> composite -> paste back; sizes must match)
        top = pil.crop((0, 0, W, 180)).convert('RGBA')
        band = Image.new('RGBA', top.size, (10, 10, 10, 130))
        pil.paste(Image.alpha_composite(top, band).convert('RGB'), (0, 0))
        d.text((60, 18), name.upper() if good else 'NO REPLY', font=font(120), fill=color)
        d.text((62, 150), 'EGO1  RISC-V + zero-DSP BNN   |   real-time gesture recognition',
               font=font(34), fill=(210, 210, 210))
        # four score bars bottom-left
        names = ['rock', 'paper', 'scissors', 'junk']
        y0 = H - 300
        for i, nm in enumerate(names):
            v = abs(scores[i]) if i < len(scores) else 0
            frac = min(1.0, v / 64.0)
            y = y0 + 48 * i
            d.text((70, y), nm.upper(), font=font(36), fill=COL[nm])
            d.rectangle([300, y + 8, 300 + int(700 * frac), y + 40], fill=COL[nm])
            d.rectangle([300, y + 8, 1000, y + 40], outline=(150, 150, 150), width=3)
            d.text((1020, y), '%+d' % scores[i] if i < len(scores) else '0', font=font(32), fill=(220, 220, 220))
        # fps badge
        d.rectangle([W - 320, H - 110, W - 40, H - 30], fill=(20, 20, 20))
        d.text((W - 300, H - 100), '%.1f fps' % fps, font=font(52), fill=(120, 255, 120))
        # target guide box: keep the hand inside this region (matches the calibration layout)
        if getattr(self, 'guide', False):
            # tighter box (user approved 2026-11-02): keep the hand near the calibrated position
            bx0, by0, bx1, by1 = int(W*0.32), int(H*0.25), int(W*0.68), int(H*0.75)
            ov = Image.new('RGBA', pil.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            od.rectangle([bx0, by0, bx1, by1], outline=(255, 230, 60, 210), width=6)
            # corner ticks
            L = 40
            for (cx0, cy0, sx, sy) in ((bx0, by0, 1, 1), (bx1, by0, -1, 1), (bx0, by1, 1, -1), (bx1, by1, -1, -1)):
                od.rectangle([min(cx0, cx0 + sx*L), cy0 - (6 if sy > 0 else 0),
                              max(cx0, cx0 + sx*L), cy0 + (6 if sy > 0 else -6)] if False else
                             [cx0 if sx > 0 else cx0 - L, cy0 - 4, cx0 + L if sx > 0 else cx0, cy0 + 4],
                             fill=(255, 230, 60, 220))
            pil = Image.alpha_composite(pil.convert('RGBA'), ov).convert('RGB')
            d = ImageDraw.Draw(pil)
            d.text((bx0 + 12, by0 + 10), '把手放在框内', font=font(30), fill=(255, 230, 60))
        return pil

    # ---------- diagnostic canvas ----------
    def render_diag(self, rgb, cls, scores, fps, thr, bits, good, flags):
        pil = Image.fromarray(rgb)
        d = ImageDraw.Draw(pil)
        name = CLASS_MAP[cls] if 0 <= cls < 4 else '?'
        color = (0, 230, 0) if good else (255, 70, 70)
        d.text((20, 10), name.upper() if good else 'NO REPLY', font=font(64), fill=color)
        d.text((20, 86), 's=%s  thr=%d  1-bit=%.2f' % (scores, thr, bits.mean()),
               font=font(22), fill=(255, 220, 0))
        d.text((20, 120), 'ok=%d bad=%d  %.1f fps  flags=0x%02x' % (self.n_ok, self.n_bad, fps, flags),
               font=font(20), fill=(210, 210, 210))
        thumb = Image.fromarray((bits * 255).astype(np.uint8)).resize((150, 150), Image.NEAREST).convert('RGB')
        pil.paste(thumb, (pil.width - 170, pil.height - 170))
        return pil


def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
    except Exception as e:
        print('[FAIL] cannot open serial %s: %s' % (PORT, e))
        return 2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('[FAIL] cannot open camera (is another program using it? close the capture tool)')
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # lock AE/AWB exactly like the capture tool (keeps the recorded scene photometrically stable)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)      # DirectShow: manual
        time.sleep(0.4)
        ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE); exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0); time.sleep(0.2)
        wb = cap.get(cv2.CAP_PROP_AUTO_WB)
        print('[camera] AE lock: auto_exposure=%.2f exposure=%.2f auto_wb=%.0f  (与采集工具一致)' % (ae, exp, wb))
    except Exception as e:
        print('[camera][WARN] AE/WB lock failed: %s' % e)
    print('[demo] mode=%s port=%s baud=%d camera=%dx%d'
          % ('REC' if REC else 'diagnostic', PORT, BAUD,
             cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    root = tk.Tk()
    Demo(root, cap, ser)
    root.mainloop()
    cap.release()
    ser.close()
    print('[demo] closed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
