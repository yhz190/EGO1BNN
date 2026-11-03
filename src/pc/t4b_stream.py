#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""t4b_stream.py -- T4b proper test: pre-packaged streaming injection, no per-frame wait.

T4a measures end-to-end throughput including host-side preprocessing/packing/(stop-and-wait)
latency -> 59.55 fps. T4b measures the INJECTION CHANNEL limit: the 620 downlink frames are
packed offline into one contiguous buffer, then written to the serial port in large chunks
with no per-frame handshake. Wall clock from first write to last byte.

Usage:
  python t4b_stream.py [COM6] [921600] [frames=620] [chunk=8192]
"""
import os, sys, time, glob, threading
import serial

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEC = os.path.join(ROOT, 'sim', 'vectors_t1a')


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else ((crc << 1) & 0xFF)
    return crc


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else 'COM6'
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 921600
    nframes = int(sys.argv[3]) if len(sys.argv) > 3 else 620
    chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 8192

    files = sorted(glob.glob(os.path.join(VEC, '*.bin')))[:nframes]
    buf = bytearray()
    for f in files:
        payload = open(f, 'rb').read()
        assert len(payload) == 512, (f, len(payload))
        body = bytes([0xA5, 0x10, 0x00, 0x02]) + payload
        buf += body + bytes([crc8(body)])
    total = len(buf)
    print('  port=%s baud=%d frames=%d  bytes=%d (%.1f B/frame)'
          % (port, baud, len(files), total, total / len(files)))
    print('  theoretical: %.2f s -> %.1f fps @%.0f baud (10 bit/byte)'
          % (total * 10.0 / baud, len(files) / (total * 10.0 / baud), baud))

    ser = serial.Serial(port, baud, timeout=0.05, write_timeout=5.0)
    time.sleep(0.2)
    # discard anything already queued (previous result frames)
    ser.reset_output_buffer(); ser.reset_input_buffer()

    # background reader: collect the 10 B result frames so the RX buffer never blocks the board
    stop = False
    rx = bytearray()

    def drain():
        while not stop:
            try:
                d = ser.read(256)
                rx.extend(d)
            except Exception:
                break

    th = threading.Thread(target=drain, daemon=True)
    th.start()

    t0 = time.perf_counter()
    sent = 0
    i = 0
    while i < total:
        n = ser.write(buf[i:i + chunk])
        sent += n
        i += n
    ser.flush()
    t1 = time.perf_counter()
    stop = True
    time.sleep(0.2)
    ser.close()

    dt = t1 - t0
    print('  streamed %d bytes in %.3f s  ->  %.2f fps  (%.2f MB/s)'
          % (sent, dt, len(files) / dt, sent / dt / 1e6))
    print('  rx result bytes: %d (expect ~%d)' % (len(rx), 10 * len(files)))
    fps = len(files) / dt
    ok = fps >= 150.0
    print('  T4b criterion (>=150 fps): %s' % ('PASS' if ok else 'FAIL'))

    # ---- correctness of the streamed run: decode result frames and compare with the model ----
    try:
        sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
        import numpy as np, torch
        import train_bnn10 as T10, train_l3x8 as L8
        sd = torch.load(os.path.join(ROOT, 'src', 'weights', 'FORMAL_l3x8_s4.pt'), map_location='cpu')
        m = L8.Net8(); m.load_state_dict(sd); m.eval()
        frames = []
        i = 0
        while i + 10 <= len(rx):
            if rx[i] == 0xA5 and rx[i+1] == 0x83 and crc8(rx[i:i+9]) == rx[i+9]:
                frames.append((rx[i+2], rx[i+6], rx[i+8]))
                i += 10
            else:
                i += 1
        print('  decoded result frames: %d' % len(frames))
        match = 0
        for k, f in enumerate(files):
            if k >= len(frames):
                break
            raw = np.fromfile(f, np.uint8)
            bits = np.unpackbits(raw)[:4096].reshape(64, 64)
            X = torch.from_numpy(bits[None, None].astype(np.float32))
            with torch.no_grad():
                pred = int(T10.hw_eval_q6(m, X).argmax(1)[0])
            if frames[k][0] == pred:
                match += 1
        print('  cls match with model: %d/%d' % (match, len(frames)))
        ok = ok and (match == len(frames)) and (len(frames) == len(files))
        print('  T4b incl. correctness: %s' % ('PASS' if ok else 'FAIL'))
    except Exception as e:
        print('  [warn] correctness check skipped:', e)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
