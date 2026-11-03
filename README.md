# EGO1 Zero-DSP BNN Gesture Accelerator

Real-time hand-gesture recognition on a low-cost FPGA (E-Elements EGO1 / Xilinx XC7A35T-1CSG324C),
built with a PicoRV32 soft core and a pure-logic binary neural network accelerator.

## What it does

Four-class gesture recognition (rock / paper / scissors / junk) on 64x64 binary images:

- PicoRV32 controls the accelerator through **Custom-0 PCPI** instructions:
  `bnn.conv` (start layer), `bnn.wait` (wait), `bnn.acc` (read result);
- 8-channel binary convolution engine: L1 3x3, L2 depthwise 3x3, L3 1x1 (to 8 ch),
  L4 4x4 majority, L5 FC 2048->4; XNOR + popcount + per-channel theta;
- **DSP48E1 usage = 0**;
- PC side performs capture, center-crop, 64x64 resize and Otsu binarization, sends a
  517-byte frame over UART (921600); the board replies with a 10-byte result frame and
  renders category / score bars / status on a 640x480@59.5Hz VGA output.

## Verified results (2026-11-02)

| Item | Value |
|---|---|
| Engine cycles | 16,911 cycles/frame @50 MHz (338.2 us, 2,957 fps) |
| End-to-end / streaming | 59.55 fps / 180.56 fps（UART 饱和；按 517 B/帧理论约 178.3 fps，差额约 1.3% 属 FTDI 波特率偏差与计时边界） |
| Resources (post-route) | LUT 4,405 (4,184 logic + 221 LUTRAM), FF 5,578, BRAM 5.0, **DSP 0**, WNS +4.136 ns |
| Accuracy | T1a in-distribution 96.07% +/- 0.47% (3 seeds); T1b same-environment 98.89% +/- 0.56% (180 frames, 2 seeds) |
| Board consistency | 180/180, 371/371, 162/162 frames match the fixed-point model exactly |
| Final demo model | onsite recalibration + spatial-illumination augmentation: 90.1% on held-out frames, board self-test 162/162 (bitstream md5 `b46b1fd6f6d256dd84c8a60fd1a91141`) |
| Protocol | downlink 517 B / uplink 10 B, CRC8 poly 0x07; weight blob 8,648 bit / 271 words (v5) |

## Repository layout

```
README.md       project intro, quick start, reproduction notes
src/            design sources: RTL, firmware, PC-side tools
sim/            testbenches, acceptance tests, simulation/tooling scripts and records
build/          reproducible build scripts and synthesis/implementation notes
board/          pin constraints, board scripts, final weight blob
data/           test vectors, reference results, design_notes/numbers.json
skill/          reusable skill package (README: scope, usage, failure conditions)
```

> **Submission archive vs. `git clone`** — `*.bit` and `*.rpt` are build products and are
> intentionally listed in `.gitignore`, so a plain `git clone` of the source repository does
> **not** contain them. The submission archive (zip/rar) still ships:
> `board/bitstream/ego1_bnn_onsite_cur16_1102.bit` (demo/recorded version),
> `board/bitstream/ego1_bnn_s1_timingfix.bit` (S1 timing-fix rebuild), and `build/*.rpt`.
> If you only have the git tree, obtain the two `.bit` files from the submission archive or
> regenerate them with `build/build_top_ascii.tcl`.
>
> **Scope** — this archive is the **code-only** package: RTL/firmware/PC tools, simulation,
> build scripts, constraints, data vectors and the reusable skill package. The design report
> PDF is submitted separately and is **not** included here. The two bitstreams and `build/*.rpt`
> are included for flashing/verification; source-only users can ignore them.

## Quick start

```powershell
# 1) Board demo: PC capture -> UART -> board inference -> VGA
src\pc\start_demo_rec.cmd

# 2) Board evidence playback (expect "371/371 frames match the model exactly")
src\pc\record_board_evidence.cmd

# 3) Onsite recalibration (new lighting / new environment, ~25 min)
#    a. capture: src\pc\start_capture.cmd
#    b. train:   python src\pc\onsite_train_aug.py --date YYYY-MM-DD --from HH:MM --to HH:MM
#    c. deploy:  powershell -File src\pc\deploy_onsite.ps1 -BlobDir <blob> -Ckpt <ckpt>
```

## Notes on the 180.56 fps figure

The streaming injection result (620 frames / 3.434 s = 180.56 fps) is slightly above the 178.3 fps bound computed for 517-byte frames at 921600 baud. The ~1.3% difference comes from the FTDI baud-rate deviation (~1%) and timing boundaries; the 620/620 result frames are correct and the conclusion (UART saturated) is unchanged.

## Reproduction notes

- Toolchain: Vivado 2022.2; Python 3.11 with numpy / torch / pyserial / opencv / Pillow.
- Build: `build/build_top_ascii.tcl` (see `build/README.md` for the required build tree).
- Data: sample test vectors are under `data/`; onsite capture data is generated locally with the
  capture scripts (the full dataset is not part of this repository).
- Numbers: acceptance values follow `report/design_notes/numbers.json` (single source of truth);
  final demo values are in its `final_demo` section.
- Paths such as `<repo>`, `<local>`, `<user-home>`, `<toolchain>` are placeholders; replace them
  with your local environment before running archived command examples.

## License

MIT License, see `LICENSE`.
