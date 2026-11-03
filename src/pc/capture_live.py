#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""capture_live.py -- guided live-capture tool for the EGO1 BNN project.

Why: T1b (cross-domain) is the current bottleneck (71.77% vs 85%). This tool builds a
real-camera dataset through EXACTLY the same preprocessing path used by training and by
the hardware frame format, so no "two-side assumption" can creep in again.

Per captured frame it writes:
  data/live/<class>/<class>_<seq:04d>.png        raw frame (auditable re-processing)
  data/live/_frames/<class>/<class>_<seq:04d>.bin  512-byte packed 1-bit frame
  data/live/_manifest.csv                       one row per frame (class, thr, bit1, pred, ...)

Controls
  1/2/3/4 : select class  (1=rock 2=paper 3=scissors 4=junk)
  SPACE   : capture one frame
  A       : auto-capture 20 frames (1 every 150 ms), then stop
  Q / ESC : quit
Live overlay shows: class | Otsu threshold | bit1 ratio | model prediction | counters.
"""
import os, sys, csv, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))

import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
import torch

import preprocess as PP
import env_stats as ES
from class_map import CLASS_MAP, CLASS_ID, assert_class_map

assert_class_map()

OUT = os.path.join(ROOT, 'src', 'data', 'live')
FRAMES = os.path.join(OUT, '_frames')
BG_DIR = os.path.join(OUT, '_env')
MANIFEST = os.path.join(OUT, '_manifest.csv')

# CJK-capable font for the overlay (cv2.putText renders non-ASCII as "?", so we draw with PIL)
_FONT_DIR = os.path.join(os.environ.get('WINDIR', ''), 'Fonts')
FONT_CANDIDATES = [os.path.join(_FONT_DIR, 'msyh.ttc'), os.path.join(_FONT_DIR, 'msyhbd.ttc'),
                   os.path.join(_FONT_DIR, 'simhei.ttf'), os.path.join(_FONT_DIR, 'simsun.ttc')]
_FONT_CACHE = {}


def _font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    f = None
    for p in FONT_CANDIDATES:
        try:
            if os.path.exists(p):
                f = ImageFont.truetype(p, size)
                break
        except Exception:
            pass
    if f is None:
        f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f

ENVSTATS = os.path.join(OUT, '_env_stats.csv')       # per-frame environment parameters
REF_JSON = os.path.join(BG_DIR, 'reference_stats.json')  # frozen environment reference
CKPT = os.path.join(ROOT, 'src', 'weights', 'B6_DA_s1.pt')

os.makedirs(BG_DIR, exist_ok=True)
for c in CLASS_MAP:
    os.makedirs(os.path.join(OUT, c), exist_ok=True)
    os.makedirs(os.path.join(FRAMES, c), exist_ok=True)


def next_seq(cls):
    """Continue numbering from max(class dir, manifest).

    The class directory can be emptied/moved (quarantine / session archive), which used to
    restart numbering at 0 and create duplicate file names in _manifest.csv
    (271 duplicates observed on 2026-11-01). Scanning the manifest as well keeps names unique.
    """
    d = os.path.join(OUT, cls)
    n = 0
    for f in os.listdir(d):
        if f.startswith(cls + '_') and f.endswith('.png'):
            try:
                n = max(n, int(f[len(cls) + 1:-4]) + 1)
            except ValueError:
                pass
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, newline='', encoding='utf-8') as fh:
                for row in csv.reader(fh):
                    if not row or row[0] == 'file':
                        continue
                    name = row[0]
                    if name.startswith(cls + '_'):
                        try:
                            n = max(n, int(name[len(cls) + 1:]) + 1)
                        except ValueError:
                            pass
        except Exception:
            pass
    return n


class App:
    def __init__(self, root, cap, predictor):
        self.root = root
        self.cap = cap
        self.pred = predictor
        self.cls = CLASS_MAP[0]
        self.auto_left = 0
        self.counts = {c: next_seq(c) for c in CLASS_MAP}
        self.last_frame = None
        self.ref = ES.load_reference(REF_JSON)
        self.env_buf = []
        self.last_env = None
        self.last_match = 1.0
        self._res = None
        self._stability = {}
        self.ema = None

        root.title('EGO1 BNN capture | 1=rock 2=paper 3=scissors 4=junk  SPACE=shot  A=auto20  B=bg20  S=LOCK-env  R=reload  C=clear-ref  Q=quit')
        self.lbl = tk.Label(root)
        self.lbl.pack()
        self.info = tk.Label(root, text='', font=('Consolas', 11), justify='left')
        self.info.pack()
        self.log = tk.Label(root, text='', font=('Consolas', 10), fg='#555')
        self.log.pack()

        root.bind('<Key>', self.on_key)
        self.update()

    # ---------------- key handling ----------------
    def on_key(self, ev):
        k = ev.keysym.lower()
        if k in ('1', '2', '3', '4'):
            self.cls = CLASS_MAP[int(k) - 1]
            self.log.config(text='class -> %s' % self.cls)
        elif k == 'space':
            self.shoot()
        elif k == 'a':
            self.auto_left = 20
        elif k == 'b':
            self.capture_background()
        elif k == 's':
            self.set_reference()
        elif k == 'r':
            self.ref = ES.load_reference(REF_JSON)
            self.log.config(text='reference reloaded: %s' % ('yes' if self.ref else 'none'))
        elif k == 'c':
            self.ref = None
            self.log.config(text='参考已清除 → 按 S 重新锁定当前环境')
        elif k in ('q', 'escape'):
            self.root.destroy()

    # ---------------- capture ----------------
    def shoot(self):
        if self.last_frame is None:
            return False
        bgr = self.last_frame.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        packed, bits, thr = PP.to_frame(img)
        bit1 = float(bits.mean())
        if bit1 < 0.55 or bit1 > 0.92:
            self.log.config(text='[WARN] bit1=%.3f out of 0.55~0.92 (RPS 训练分布) — 请调光照/背景/距离，仍可保存' % bit1)
        if set(np.unique(bits).tolist()) - {0, 1}:
            self.log.config(text='[REJECT] non-binary frame')
            return False

        seq = self.counts[self.cls]
        self.counts[self.cls] = seq + 1
        name = '%s_%04d' % (self.cls, seq)
        # cv2.imwrite cannot handle non-ASCII paths (src) -> silently fails; use PIL
        Image.fromarray(rgb).save(os.path.join(OUT, self.cls, name + '.png'))
        with open(os.path.join(FRAMES, self.cls, name + '.bin'), 'wb') as fh:
            fh.write(packed)
        Image.fromarray((bits * 255).astype(np.uint8)).save(
            os.path.join(FRAMES, self.cls, name + '_bin64.png'))
        Image.fromarray(np.asarray(img.convert('L').resize((64, 64)))).save(
            os.path.join(FRAMES, self.cls, name + '_gray64.png'))

        pred, scores = self.pred(bits)
        new = not os.path.exists(MANIFEST)
        with open(MANIFEST, 'a', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(['file', 'class', 'thr', 'bit1', 'pred', 'scores', 'ts'])
            w.writerow([name, self.cls, thr, '%.4f' % bits.mean(), pred,
                        ' '.join(str(int(s)) for s in scores), int(time.time())])
        self.log.config(text='saved %s  (pred=%s)  counts=%s'
                        % (name, pred, {c: self.counts[c] for c in CLASS_MAP}))
        if self.last_env is not None:
            new2 = not os.path.exists(ENVSTATS)
            with open(ENVSTATS, 'a', newline='', encoding='utf-8') as fh2:
                w2 = csv.writer(fh2)
                if new2:
                    w2.writerow(['file', 'class', 'area', 'cy', 'cx', 'centroid', 'border',
                                 'bg_level', 'bg_std', 'thr_ratio', 'thr', 'bit1',
                                 'env_ok', 'pose_ok'])
                e = self.last_env
                w2.writerow([name, self.cls, '%.4f' % e['area'], '%.4f' % e['cy'],
                             '%.4f' % e['cx'], '%.4f' % e['centroid'], '%.4f' % e['border'],
                             '%.1f' % e['bg_level'], '%.1f' % e['bg_std'],
                             '%.4f' % e['thr_ratio'], '%.1f' % e['thr'], '%.4f' % e['bit1'],
                             int(bool(self._res and self._res['env_ok'])),
                             int(bool(self._res and self._res['pose_ok']))])
        return True

    # ---------------- environment reference ----------------
    def set_reference(self):
        """Lock the ENVIRONMENT (lighting/camera/background), not the hand pose.

        Press S while slowly moving through several poses of the class you are shooting: the
        reference then stores mean AND std, so natural wobble stays inside the pose band and
        only real environment changes are reported as DRIFT."""
        if len(self.env_buf) < 8:
            self.log.config(text='[WAIT] show the scene for ~1 s, then press S (poses may vary)')
            return
        keys = self.env_buf[0].keys()
        recent = self.env_buf[-40:]
        st = {k: {'mean': float(np.mean([e[k] for e in recent])),
                  'std': float(np.std([e[k] for e in recent]))} for k in keys}
        ES.save_reference(st, REF_JSON)
        self.ref = st
        band = lambda f, fl: (st[f]['mean'] - max(fl, 3 * st[f]['std']),
                             st[f]['mean'] + max(fl, 3 * st[f]['std']))
        l1 = band('bg_level', 8.0); l2 = band('bg_std', 6.0); l3 = band('thr_ratio', 0.05)
        self.log.config(text='环境已锁定（姿态自由）: 背景亮度 %.0f~%.0f, 背景纹理 %.0f~%.0f, 光照对比 %.2f~%.2f'
                            % (l1[0], l1[1], l2[0], l2[1], l3[0], l3[1]))

    # ---------------- background capture ----------------
    def capture_background(self, n=20):
        """Save n empty-scene frames (clear the desk first!). Used by preprocess3 background
        subtraction and by the domain probe to detect environment drift."""
        stamp = time.strftime('%Y%m%d_%H%M%S')
        saved = 0
        for i in range(n):
            ok, bgr = self.cap.read()
            if not ok:
                continue
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(
                os.path.join(BG_DIR, 'bg_%s_%02d.png' % (stamp, i)))
            saved += 1
            time.sleep(0.05)
        self.bg_name = 'bg_%s' % stamp
        self.log.config(text='saved %d background frames -> %s/bg_%s_*.png  (场景已清空?)'
                            % (saved, BG_DIR, stamp))

    # ---------------- main loop ----------------
    def update(self):
        ok, bgr = self.cap.read()
        if ok:
            self.last_frame = bgr
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            packed, bits, thr = PP.to_frame(Image.fromarray(rgb))
            pred, scores = self.pred(bits)
            gray = np.asarray(Image.fromarray(rgb).convert('L'), np.uint8)
            self.last_env = ES.frame_params(gray, bits, thr)
            self.env_buf.append(self.last_env)
            if len(self.env_buf) > 50:
                self.env_buf.pop(0)
            _res = ES.assess(self.last_env, self.ref) if self.ref is not None else None
            # EMA: display values are smoothed so hand wobble does not make the text jump
            if self.ema is None:
                self.ema = dict(self.last_env)
            else:
                a = 0.25
                for k in list(self.ema.keys()):
                    if k in self.last_env:
                        self.ema[k] = (1 - a) * self.ema[k] + a * self.last_env[k]
            self._stability = ES.window_spread(self.env_buf[-20:]) if len(self.env_buf) >= 5 else {}
            # judge the WINDOW MEDIAN, not the single frame: gesture motion cannot flicker the verdict
            _win = ES.window_median(self.env_buf[-20:]) if self.ref is not None else None
            _res = ES.assess(_win, self.ref) if _win else None
            self._res = _res
            self._win = _win

            # ---- overlay drawn with PIL (Chinese renders correctly; cv2 would show "?") ----
            pil = Image.fromarray(rgb)
            d = ImageDraw.Draw(pil)
            colors = {'ok': (0, 230, 0), 'bad': (255, 70, 70), 'warn': (255, 210, 0),
                      'info': (215, 215, 215), 'hint': (255, 210, 0)}
            d.text((8, 8), '%s   预测=%s   thr=%d   1-bit=%.2f'
                   % (self.cls.upper(), pred, thr, bits.mean()), font=_font(22), fill=(0, 255, 0))
            y = 42
            for text, kind in ES.status_lines(self._win if getattr(self, '_win', None) else self.ema,
                                             self.ref, _res):
                d.text((8, y), text, font=_font(17), fill=colors.get(kind, colors['info']))
                y += 26
            st = self._stability or {}
            d.text((8, y), '窗口抖动: 手面积%.1f%%  背景亮度%.1f%%   %s'
                   % (st.get('area', 0) * 100, st.get('bg_level', 0) * 100,
                      '稳定' if st.get('area', 0) < 0.05 else '手在动（正常，不影响标定）'),
                   font=_font(15), fill=(190, 190, 190))
            y += 24
            d.text((8, y), '流程: 1)清空桌面按B拍空背景 2)摆好机位/灯光 3)做3~4个同类手势按S锁环境 4)按A连拍',
                   font=_font(15), fill=(170, 200, 255))
            y += 24
            d.text((8, y), '计数: ' + '  '.join('%s=%d' % (c, self.counts[c]) for c in CLASS_MAP),
                   font=_font(15), fill=(255, 255, 0))
            thumb = Image.fromarray((bits * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST).convert('RGB')
            pil.paste(thumb, (pil.width - 136, pil.height - 136))
            self.tk = ImageTk.PhotoImage(pil)
            self.lbl.config(image=self.tk)
            self.info.config(text='class=%s  1-bit=%.3f  pred=%s  auto_left=%d  env=%s'
                                  % (self.cls, bits.mean(), pred, self.auto_left,
                                     ('OK' if (_res and _res['env_ok']) else ('DRIFT' if _res else '-'))))
        if self.auto_left > 0:
            if self.shoot():
                self.auto_left -= 1
            self.root.after(150, self.update)
        else:
            self.root.after(40, self.update)


def main():
    # predictor: exact integer forward on the 8-channel model
    pred_fn = None
    if os.path.exists(CKPT):
        import train_l3x8 as L8
        sd = torch.load(CKPT, map_location='cpu')
        m = L8.Net8(); m.load_state_dict(sd); m.eval()

        def pred_fn(bits):                       # bits: 64x64 uint8 0/1
            X = torch.from_numpy(bits[None, None, :, :].astype(np.float32))
            with torch.no_grad():
                _, ex_int = L8.hw_eval8(m, X)
            p = int(ex_int.argmax(1)[0])
            return CLASS_MAP[p], ex_int[0].tolist()
    else:
        def pred_fn(bits):
            return 'n/a', [0, 0, 0, 0]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('[FAIL] cannot open camera'); return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print('[camera] %dx%d' % (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # ---- lock auto-exposure / auto-white-balance -------------------------------------
    # Root cause of a real data loss (2026-11-01 17:46): the webcam's auto-exposure changed the
    # background level by +30~45% when the hand covered more/less of the frame, turning one
    # capture session into two "domains". Locking AE/WB keeps the environment constant.
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)      # DirectShow: 0.25 == manual mode
        time.sleep(0.4)
        ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        time.sleep(0.2)
        wb = cap.get(cv2.CAP_PROP_AUTO_WB)
        print('[camera] AE lock: auto_exposure=%.2f exposure=%.2f auto_wb=%.0f' % (ae, exp, wb))
        if ae > 0.5:
            print('[camera][WARN] 自动曝光可能未锁上（auto_exposure=%.2f）；若数据仍有环境漂移，'
                  '请在相机驱动/系统设置里手动固定曝光' % ae)
    except Exception as e:
        print('[camera][WARN] AE/WB lock failed: %s' % e)

    # baseline: report the environment fingerprint for 20 frames so drift is visible immediately
    try:
        _vals = []
        for _ in range(20):
            ok, _b = cap.read()
            if not ok:
                continue
            _g = np.asarray(Image.fromarray(cv2.cvtColor(_b, cv2.COLOR_BGR2RGB)).convert('L'), np.uint8)
            _, _bits, _thr = PP.to_frame(Image.fromarray(cv2.cvtColor(_b, cv2.COLOR_BGR2RGB)))
            _p = None
            try:
                import env_stats as _ES
                _p = _ES.frame_params(_g, _bits, _thr)
            except Exception:
                pass
            if _p:
                _vals.append(_p['bg_level'])
        if _vals:
            import statistics as _st
            print('[camera] 环境基线: 背景亮度 中位=%.0f 标准差=%.1f （若标准差>5 说明曝光仍在漂）'
                  % (_st.median(_vals), _st.pstdev(_vals)))
    except Exception as e:
        print('[camera] baseline skipped: %s' % e)
    print('[ckpt] %s' % (os.path.basename(CKPT) if os.path.exists(CKPT) else 'none'))
    print('[out] %s' % OUT)

    if os.path.exists(REF_JSON):
        _age = (time.time() - os.path.getmtime(REF_JSON)) / 60.0
        print('[ref] 已载入环境参考 %s（%.1f 分钟前生成）；若光照与参考不符，按 S 重锁、C 清除'
              % (os.path.basename(REF_JSON), _age))
    else:
        print('[ref] 无参考 → 摆好机位后按 S 锁定')

    root = tk.Tk()
    App(root, cap, pred_fn)
    root.mainloop()
    cap.release()
    print('[done] manifest: %s' % MANIFEST)
    return 0


if __name__ == '__main__':
    sys.exit(main())
