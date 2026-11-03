// tb_top_bnn.v -- top-level self-checking testbench for top_bnn (v2.2-patch2)
//
// Checks
//   1. PicoRV32 boots from IMEM and runs the real firmware (fw.hex), and the CPU can
//      write + read back an SFR (MARGIN 0x0001_0028, pattern 0xA55A).  The historic
//      "SFR write leaks into IMEM" bug is checked three ways: an IMEM write counter
//      that must stay 0, a snapshot of imem[0], and the fact that the firmware only
//      works if its own instructions stay intact.
//   2. The CPU launches one full frame through PCPI bnn.conv, blocks in bnn.wait and
//      reads the 4 scores with bnn.acc; every value is compared with the Verilog
//      array reference below (independent of the DUT: the weights reach the engine
//      only through the 271 word blob, so a broken unpack shows up here).
//   3. An illegal Custom-0 instruction (funct3=7) must be answered with
//      pcpi_ready=1 / pcpi_wr=0 and the CPU must keep executing (the firmware
//      continues to the polling loop and later runs the frame).
//   4. The frame length must be exactly 16,911 engine cycles (per-layer census).
//   5. The 10 B result frame on the UART TX pin is decoded and checked.
//   6. A/B double buffering: frame B is streamed in *during* frame A; no write may hit
//      the bank being read, and both frames must produce their own correct scores.
//   7. Real v5 blob: a second instance (dut2, WB_INIT=1) boots from the actual
//      weights.hex / fw.hex on disk.  The TB re-parses the file (271 lines, ID, CRC16,
//      beat-major FC structural check) and runs frame C through the same reference
//      arrays; all 4 scores must match bit for bit.
//
// Stimulus
//   * synthetic 64x64 frame built in the TB, sent as a real 517 B downlink frame over
//     pc_uart_rxd (a corrupted-CRC frame is sent first and must NOT raise frame_ready)
//   * dut : LFSR weights / theta, packed into the v5 blob layout (8,648 bit -> 271
//     words) and written straight into dut.wbmem before reset release (WB_INIT=0)
//   * dut2: the real v5 blob from board/weights/hw/weights.hex (beat-major FC)
//   * BAUD_DIV is overridden to 8 (the RTL default 434 = 115200 baud @50 MHz)
//
// Run (see sim/20261101_beatmajor/):
//   xvlog -i rtl -i rtl/bnn <rtl files> tb_top_bnn.v
//   xelab tb_top_bnn -s sim -L unisims_ver
//   xsim sim -runall        (weights.hex + fw.hex must be in the run directory)
`timescale 1ns / 1ps

module tb_top_bnn;

    // ================================================================ clock / reset
    reg clk100, rst_n;
    initial clk100 = 1'b0;
    always #5 clk100 = ~clk100;              // 100 MHz board clock

    localparam integer BAUD_DIV = 8;         // TB only (RTL default: 434)
    localparam integer TB_BIT   = 2*BAUD_DIV; // clk100 cycles per UART bit (50 MHz domain)

    localparam integer EXP_L1 = 4228, EXP_BAR = 1, EXP_L2 = 4228,
                       EXP_L3 = 4098, EXP_L4 = 4098, EXP_L5 = 257, EXP_DONE = 1,
                       EXP_FRAME = 16911;

    reg  rxd;
    wire txd;
    wire [3:0] led;

    top_bnn #(
        .BAUD_DIV(BAUD_DIV), .FPS_WIN(20000), .WB_INIT(0), .MODEL_ID(16'h0001)
    ) dut (
        .sys_clk_100m(clk100), .sys_rst_n(rst_n),
        .pc_uart_rxd(rxd), .pc_uart_txd(txd),
        .led(led), .vga_hs(), .vga_vs(), .vga_rgb()
    );

    // ================================================================ second DUT: real v5 blob
    // dut2 boots from the *real* files on disk: $readmemh fw.hex + weights.hex
    // (WB_INIT=1), i.e. the exact bit stream a board build would use.  It is used for
    // the "gen_weights v5 -> engine -> 4 scores" end-to-end check, while dut keeps the
    // LFSR / D5 scenarios.  The two never share UART pins.
    reg  rxd2;
    wire txd2;
    wire [3:0] led2;

    top_bnn #(
        .BAUD_DIV(BAUD_DIV), .FPS_WIN(20000), .WB_INIT(1), .MODEL_ID(16'h0001)
    ) dut2 (
        .sys_clk_100m(clk100), .sys_rst_n(rst_n),
        .pc_uart_rxd(rxd2), .pc_uart_txd(txd2),
        .led(led2), .vga_hs(), .vga_vs(), .vga_rgb()
    );

    // ================================================================ scoreboard
    integer errors, checks;

    task chk;
        input [8*64-1:0] name;
        input integer    got;
        input integer    exp;
        begin
            if (got !== exp) begin
                $display("[FAIL] %0s : got %0d (0x%0h), exp %0d (0x%0h)", name, got, got, exp, exp);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s : %0d", name, got);
                checks = checks + 1;
            end
        end
    endtask

    task chk32;
        input [8*64-1:0] name;
        input [31:0]     got;
        input [31:0]     exp;
        begin
            if (got !== exp) begin
                $display("[FAIL] %0s : got %08x, exp %08x", name, got, exp);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s : %08x", name, got);
                checks = checks + 1;
            end
        end
    endtask

    // ================================================================ stimulus data
    reg [71:0]   W1, W2;
    reg [63:0]   W3;
    reg [2047:0] WFA [0:3];
    reg [3:0]    TH1 [0:7], TH2 [0:7], TH3 [0:7];
    reg [6:0]    TH4 [0:7];
    reg signed [15:0] THFC [0:3];

    reg          frm [0:4095];
    reg          frm_a [0:4095];
    reg [7:0]    payload [0:511];
    reg [7:0]    payload_a [0:511];
    reg [7:0]    payload_b [0:511];
    reg [31:0]   blob [0:270];

    // ---- real v5 blob (271 words read from weights.hex) and its TB-side parse ----
    reg [31:0]   real_blob [0:270];
    reg [71:0]   W1R, W2R;
    reg [63:0]   W3R;
    reg [2047:0] WFA_R [0:3];
    reg [3:0]    TH1R [0:7], TH2R [0:7], TH3R [0:7];
    reg [6:0]    TH4R [0:7];
    reg signed [15:0] THFCR [0:3];
    integer      n_blob_lines;

    // frame C (the real-weight inference on dut2) reference
    integer      refc_acc   [0:3];
    integer      refc_score [0:3];
    integer      refc_cls;

    reg [7:0]    ref_a1 [0:4095];
    reg [7:0]    ref_a2 [0:4095];
    reg [7:0]    ref_a3 [0:4095];
    reg [7:0]    ref_feat [0:255];
    integer      ref_acc   [0:3];
    integer      ref_score [0:3];
    integer      exp_cls;
    integer      refa_acc   [0:3];
    integer      refa_score [0:3];
    integer      refa_cls;
    integer      refb_acc   [0:3];
    integer      refb_score [0:3];
    integer      refb_cls;

    reg [31:0]   lfsr;
    task lstep;
        begin
            lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
        end
    endtask

    task gen_stimulus;
        integer i, c, t, tmp;
        reg [95:0]  wtmp;
        begin
            lfsr = 32'hC0FF_EE01;
            // ---- L1 / L2 weights (8 ch x 9 tap) ----
            for (i = 0; i < 3; i = i + 1) begin lstep; wtmp[i*32 +: 32] = lfsr; end
            W1 = wtmp[71:0];
            for (i = 0; i < 3; i = i + 1) begin lstep; wtmp[i*32 +: 32] = lfsr; end
            W2 = wtmp[71:0];
            for (i = 0; i < 2; i = i + 1) begin lstep; wtmp[i*32 +: 32] = lfsr; end
            W3 = wtmp[63:0];
            // ---- L5 weights: 4 classes x 2048 bit ----
            for (c = 0; c < 4; c = c + 1) WFA[c] = 2048'd0;
            for (c = 0; c < 4; c = c + 1)
                for (i = 0; i < 64; i = i + 1) begin
                    lstep;
                    WFA[c][i*32 +: 32] = lfsr;
                end
            // ---- theta: keep the activations mixed (0..7 of 9 / 0..15 of 16) ----
            for (c = 0; c < 8; c = c + 1) begin
                lstep; TH1[c] = lfsr[2:0];
                lstep; TH2[c] = lfsr[2:0];
                lstep; TH3[c] = lfsr[2:0];
                lstep; TH4[c] = {3'd0, lfsr[3:0]};            // 0..15, 7 bit field
            end
            for (c = 0; c < 4; c = c + 1) begin
                lstep;
                THFC[c] = $signed({3'b0, lfsr[12:0]}) - 16'sd2048;   // -2048..+2047
            end
            // ---- synthetic frame (deterministic, differs from tb_pipeline) ----
            for (i = 0; i < 4096; i = i + 1) begin
                tmp = 0;
                if ((i/64 >= 6) && (i/64 <= 30) && (i%64 >= 10) && (i%64 <= 20)) tmp = 1;
                if ((i/64) == (i%64)) tmp = 1;
                if (((i%64) % 9) == 0 && ((i/64) % 5) == 0) tmp = 1;
                if ((((i/64)*3 + (i%64)*5) % 13) == 0) tmp = 1;
                frm[i] = (tmp != 0);
            end
            for (i = 0; i < 4096; i = i + 1) frm_a[i] = frm[i];
        end
    endtask

    // ---- pack the stimulus into the v5 blob (bit i -> word i/32, bit i%32) ----
    task setbit;
        input integer idx;
        input integer v;
        begin
            if (v != 0) blob[idx/32] = blob[idx/32] | (32'd1 << (idx % 32));
            else        blob[idx/32] = blob[idx/32] & ~(32'd1 << (idx % 32));
        end
    endtask

    task build_blob;
        integer c, t, f, q, j;
        begin
            for (c = 0; c < 271; c = c + 1) blob[c] = 32'd0;
            for (c = 0; c < 8; c = c + 1)
                for (t = 0; t < 9; t = t + 1)
                    setbit(0 + c*9 + t, (W1 >> (c*9 + t)) & 1);
            for (c = 0; c < 8; c = c + 1)
                for (t = 0; t < 9; t = t + 1)
                    setbit(72 + c*9 + t, (W2 >> (c*9 + t)) & 1);
            for (c = 0; c < 8; c = c + 1)
                for (t = 0; t < 8; t = t + 1)
                    setbit(144 + c*8 + t, (W3 >> (c*8 + t)) & 1);
            // v5: FC field is beat-major -- blob bit 208+32q+8c+j = WFA[c][8q+j]
            for (q = 0; q < 256; q = q + 1)
                for (c = 0; c < 4; c = c + 1)
                    for (j = 0; j < 8; j = j + 1)
                        setbit(208 + 32*q + 8*c + j, (WFA[c] >> (8*q + j)) & 1);
            for (c = 0; c < 8; c = c + 1) begin
                for (t = 0; t < 4; t = t + 1) setbit(8400 + c*4 + t, (TH1[c] >> t) & 1);
                for (t = 0; t < 4; t = t + 1) setbit(8432 + c*4 + t, (TH2[c] >> t) & 1);
                for (t = 0; t < 4; t = t + 1) setbit(8464 + c*4 + t, (TH3[c] >> t) & 1);
                for (t = 0; t < 7; t = t + 1) setbit(8496 + c*7 + t, (TH4[c] >> t) & 1);
            end
            for (c = 0; c < 4; c = c + 1)
                for (t = 0; t < 16; t = t + 1)
                    setbit(8552 + c*16 + t, (THFC[c] >> t) & 1);
            for (t = 0; t < 16; t = t + 1) setbit(8616 + t, (16'h0001 >> t) & 1);
            // ID at [8616,8632); CRC16 at [8632,8648) is not checked by the RTL
        end
    endtask

    task pack_payload;
        integer i, j;
        reg [7:0] b;
        begin
            for (i = 0; i < 512; i = i + 1) begin
                b = 8'd0;
                for (j = 0; j < 8; j = j + 1)
                    b = b | (frm[i*8 + j] << (7 - j));       // MSB first, row major
                payload[i] = b;
            end
        end
    endtask

    // ================================================================ real v5 blob parse
    // Blob bit i -> word i/32, bit i%32 (LSB first).  The TB re-derives every field
    // from the *file* with its own code; nothing is read out of dut2.
    function integer blob_bit;
        input integer idx;
        begin
            blob_bit = (real_blob[idx/32] >> (idx % 32)) & 1;
        end
    endfunction

    function [15:0] crc16_byte;
        input [15:0] c;
        input [7:0]  d;
        integer i;
        reg [15:0] x;
        begin
            x = c ^ {d, 8'h00};
            for (i = 0; i < 8; i = i + 1)
                x = x[15] ? ((x << 1) ^ 16'h1021) : ((x << 1) & 16'hFFFF);
            crc16_byte = x;
        end
    endfunction

    task parse_real_blob;
        integer c, t, j, q, i, k, b, nbad, idv;
        reg [15:0] crc_calc, crc_stored;
        reg [31:0] expw, gotw;
        reg [8*256-1:0] linebuf;
        integer fd;
        begin
            $readmemh("weights.hex", real_blob);

            // ---- 271 lines (the RTL blob length is fixed at 271 words) ----
            fd = $fopen("weights.hex", "r");
            n_blob_lines = 0;
            linebuf = 0;
            if (fd == 0) begin
                n_blob_lines = -1;
            end else begin
                while ($fgets(linebuf, fd) != 0) n_blob_lines = n_blob_lines + 1;
                $fclose(fd);
            end

            // ---- ID / CRC16 (the only fields with a fixed expected value) ----
            idv = 0;
            for (k = 0; k < 16; k = k + 1) idv = idv | (blob_bit(8616 + k) << k);
            crc_stored = 0;
            for (k = 0; k < 16; k = k + 1) crc_stored = crc_stored | (blob_bit(8632 + k) << k);
            crc_calc = 16'hFFFF;
            for (i = 0; i < 1077; i = i + 1) begin
                b = 0;
                for (k = 0; k < 8; k = k + 1) b = b | (blob_bit(i*8 + k) << k);
                crc_calc = crc16_byte(crc_calc, b[7:0]);
            end

            // ---- weights (v5 beat-major FC) ----
            for (c = 0; c < 8; c = c + 1)
                for (t = 0; t < 9; t = t + 1) begin
                    W1R[c*9 + t] = blob_bit(0 + c*9 + t);
                    W2R[c*9 + t] = blob_bit(72 + c*9 + t);
                end
            for (c = 0; c < 8; c = c + 1)
                for (t = 0; t < 8; t = t + 1)
                    W3R[c*8 + t] = blob_bit(144 + c*8 + t);
            for (c = 0; c < 4; c = c + 1) WFA_R[c] = 2048'd0;
            for (q = 0; q < 256; q = q + 1)
                for (c = 0; c < 4; c = c + 1)
                    for (j = 0; j < 8; j = j + 1)
                        WFA_R[c][8*q + j] = blob_bit(208 + 32*q + 8*c + j);

            // ---- theta ----
            for (c = 0; c < 8; c = c + 1) begin
                for (t = 0; t < 4; t = t + 1) begin
                    TH1R[c][t] = blob_bit(8400 + c*4 + t);
                    TH2R[c][t] = blob_bit(8432 + c*4 + t);
                    TH3R[c][t] = blob_bit(8464 + c*4 + t);
                end
                for (t = 0; t < 7; t = t + 1)
                    TH4R[c][t] = blob_bit(8496 + c*7 + t);
            end
            for (c = 0; c < 4; c = c + 1)
                for (t = 0; t < 16; t = t + 1)
                    THFCR[c][t] = blob_bit(8552 + c*16 + t);

            // ---- structural check: every beat word in the file really is {b3,b2,b1,b0} ----
            nbad = 0;
            for (q = 0; q < 256; q = q + 1) begin
                expw = 32'd0;
                for (c = 0; c < 4; c = c + 1)
                    for (j = 0; j < 8; j = j + 1)
                        expw = expw | (WFA_R[c][8*q + j] << (8*c + j));
                gotw = (real_blob[6 + q] >> 16) | (real_blob[7 + q] << 16);
                if (gotw !== expw) nbad = nbad + 1;
            end

            chk("v5 weights.hex: 271 lines", n_blob_lines, 271);
            chk32("v5 weights.hex: ID field", {16'd0, idv[15:0]}, 32'h0000_0001);
            chk32("v5 weights.hex: CRC16 recomputed == stored",
                  {16'd0, crc_calc}, {16'd0, crc_stored});
            chk("v5 FC beat-major: all 256 beat words == {b3,b2,b1,b0}", nbad, 0);
            $display("[tb] v5 blob parsed: CRC 0x%04X, W1=%018h W2=%018h W3=%016h",
                     crc_calc, W1R, W2R, W3R);
            $display("[tb] v5 real theta: TH1=%0d/%0d/%0d/%0d TH2=%0d/%0d/%0d/%0d",
                     TH1R[0], TH1R[1], TH1R[2], TH1R[3], TH2R[0], TH2R[1], TH2R[2], TH2R[3]);
            $display("[tb] v5 real bias_q: %0d %0d %0d %0d",
                     THFCR[0], THFCR[1], THFCR[2], THFCR[3]);
        end
    endtask

    // ================================================================ dut2 UART stimulus
    // (send_byte2 / send_frame2 are declared after crc8 / send_frame below)

    // ================================================================ reference model
    function integer classify;
        input integer q0, q1, q2, q3, margin;
        integer cls, top1, top2;
        begin
            if ((q3 > q0) && (q3 > q1) && (q3 > q2)) cls = 3;
            else begin
                if ((q0 >= q1) && (q0 >= q2)) cls = 0;
                else if (q1 >= q2)           cls = 1;
                else                         cls = 2;
                top1 = (cls == 0) ? q0 : ((cls == 1) ? q1 : q2);
                top2 = (cls == 0) ? ((q1 > q2) ? q1 : q2)
                     : (cls == 1) ? ((q0 > q2) ? q0 : q2)
                                  : ((q0 > q1) ? q0 : q1);
                if ((top1 - top2) < margin) cls = 3;
            end
            classify = cls;
        end
    endfunction

    task build_ref;
        integer p, r, c, ch, k, ky, kx, rr, cc, b, s, by, bx, B, ry, rx, f, fb;
        begin
            // ---------------- L1 ----------------
            for (p = 0; p < 4096; p = p + 1) begin
                r = p / 64;
                c = p % 64;
                for (ch = 0; ch < 8; ch = ch + 1) begin
                    s = 0;
                    for (ky = 0; ky < 3; ky = ky + 1)
                        for (kx = 0; kx < 3; kx = kx + 1) begin
                            rr = r + ky - 1;
                            cc = c + kx - 1;
                            b = 0;
                            if ((rr >= 0) && (rr < 64) && (cc >= 0) && (cc < 64))
                                b = frm[rr*64 + cc];
                            if (b == ((W1 >> (ch*9 + ky*3 + kx)) & 1)) s = s + 1;
                        end
                    ref_a1[p][ch] = (s > TH1[ch]);
                end
            end
            // ---------------- L2 (depthwise) ----------------
            for (p = 0; p < 4096; p = p + 1) begin
                r = p / 64;
                c = p % 64;
                for (ch = 0; ch < 8; ch = ch + 1) begin
                    s = 0;
                    for (ky = 0; ky < 3; ky = ky + 1)
                        for (kx = 0; kx < 3; kx = kx + 1) begin
                            rr = r + ky - 1;
                            cc = c + kx - 1;
                            b = 0;
                            if ((rr >= 0) && (rr < 64) && (cc >= 0) && (cc < 64))
                                b = (ref_a1[rr*64 + cc] >> ch) & 1;
                            if (b == ((W2 >> (ch*9 + ky*3 + kx)) & 1)) s = s + 1;
                        end
                    ref_a2[p][ch] = (s > TH2[ch]);
                end
            end
            // ---------------- L3 1x1, 8 -> 8 ----------------
            for (p = 0; p < 4096; p = p + 1)
                for (k = 0; k < 8; k = k + 1) begin
                    s = 0;
                    for (ch = 0; ch < 8; ch = ch + 1)
                        if (ref_a2[p][ch] == ((W3 >> (k*8 + ch)) & 1)) s = s + 1;
                    ref_a3[p][k] = (s > TH3[k]);
                end
            // ---------------- L4 4x4 majority ----------------
            for (by = 0; by < 16; by = by + 1)
                for (bx = 0; bx < 16; bx = bx + 1) begin
                    B = by*16 + bx;
                    for (ch = 0; ch < 8; ch = ch + 1) begin
                        s = 0;
                        for (ry = 0; ry < 4; ry = ry + 1)
                            for (rx = 0; rx < 4; rx = rx + 1)
                                s = s + ref_a3[(4*by + ry)*64 + (4*bx + rx)][ch];
                        ref_feat[B][ch] = (s >= (TH4[ch] & 7'h1F));
                    end
                end
            // ---------------- L5 FC 2048 -> 4, Q6 ----------------
            for (k = 0; k < 4; k = k + 1) begin
                s = 0;
                for (f = 0; f < 2048; f = f + 1) begin
                    fb = ref_feat[f % 256][f / 256];
                    if (fb == WFA[k][f]) s = s + 1;
                end
                ref_acc[k]   = s;
                ref_score[k] = (2*s - 2048) * 64 + THFC[k];
            end
            exp_cls = classify(ref_score[0], ref_score[1], ref_score[2], ref_score[3], 1536);
        end
    endtask

    // ================================================================ UART stimulus
    task send_byte;
        input [7:0] b;
        integer i;
        begin
            rxd = 1'b0;
            repeat (TB_BIT) @(posedge clk100);
            for (i = 0; i < 8; i = i + 1) begin
                rxd = b[i];
                repeat (TB_BIT) @(posedge clk100);
            end
            rxd = 1'b1;
            repeat (TB_BIT) @(posedge clk100);
        end
    endtask

    function [7:0] crc8;
        input [7:0] c, d;
        integer i;
        begin
            crc8 = c ^ d;
            for (i = 0; i < 8; i = i + 1)
                crc8 = crc8[7] ? ((crc8 << 1) ^ 8'h07) : (crc8 << 1);
        end
    endfunction

    task send_frame;
        input integer corrupt;
        integer i;
        reg [7:0] crc;
        begin
            crc = 8'h00;
            send_byte(8'hA5); crc = crc8(crc, 8'hA5);
            send_byte(8'h10); crc = crc8(crc, 8'h10);
            send_byte(8'h00); crc = crc8(crc, 8'h00);
            send_byte(8'h02); crc = crc8(crc, 8'h02);
            for (i = 0; i < 512; i = i + 1) begin
                send_byte(payload[i]);
                crc = crc8(crc, payload[i]);
            end
            if (corrupt != 0) send_byte(crc ^ 8'hFF);
            else              send_byte(crc);
            repeat (TB_BIT*4) @(posedge clk100);
        end
    endtask

    // ---- dut2 UART stimulus (same 517 B frame format, different pins) ----
    task send_byte2;
        input [7:0] b;
        integer i;
        begin
            rxd2 = 1'b0;
            repeat (TB_BIT) @(posedge clk100);
            for (i = 0; i < 8; i = i + 1) begin
                rxd2 = b[i];
                repeat (TB_BIT) @(posedge clk100);
            end
            rxd2 = 1'b1;
            repeat (TB_BIT) @(posedge clk100);
        end
    endtask

    task send_frame2;
        input integer corrupt;
        integer i;
        reg [7:0] crc;
        begin
            crc = 8'h00;
            send_byte2(8'hA5); crc = crc8(crc, 8'hA5);
            send_byte2(8'h10); crc = crc8(crc, 8'h10);
            send_byte2(8'h00); crc = crc8(crc, 8'h00);
            send_byte2(8'h02); crc = crc8(crc, 8'h02);
            for (i = 0; i < 512; i = i + 1) begin
                send_byte2(payload[i]);
                crc = crc8(crc, payload[i]);
            end
            if (corrupt != 0) send_byte2(crc ^ 8'hFF);
            else              send_byte2(crc);
            repeat (TB_BIT*4) @(posedge clk100);
        end
    endtask

    // D5: split a 517 B frame around a running inference.  prefix() sends the header
    // and payload bytes 0..n-1; suffix(n) sends payload bytes n..511 and the CRC (the
    // CRC is recomputed locally over the whole frame, so the two halves stay coherent).
    task send_frame_prefix;
        input integer n;
        integer i;
        reg [7:0] crc;
        begin
            crc = 8'h00;
            send_byte(8'hA5); crc = crc8(crc, 8'hA5);
            send_byte(8'h10); crc = crc8(crc, 8'h10);
            send_byte(8'h00); crc = crc8(crc, 8'h00);
            send_byte(8'h02); crc = crc8(crc, 8'h02);
            for (i = 0; i < n; i = i + 1) begin
                send_byte(payload[i]);
                crc = crc8(crc, payload[i]);
            end
        end
    endtask

    task send_frame_suffix;
        input integer n;
        integer i;
        reg [7:0] crc;
        begin
            crc = 8'h00;
            crc = crc8(crc, 8'hA5);
            crc = crc8(crc, 8'h10);
            crc = crc8(crc, 8'h00);
            crc = crc8(crc, 8'h02);
            for (i = 0; i < 512; i = i + 1) crc = crc8(crc, payload[i]);
            for (i = n; i < 512; i = i + 1) send_byte(payload[i]);
            send_byte(crc);
            repeat (TB_BIT*4) @(posedge clk100);
        end
    endtask

    // ================================================================ monitors
    integer n_st [0:7];
    integer n_conv, n_wait, n_acc, n_ill, n_ill_bad, n_unknown_f3, n_wait_stall;
    reg [31:0] acc_val [0:3];
    reg        acc_got [0:3];

    integer n_sfr_wr, n_imem_wr, n_dmem_wr, n_margin_rd, n_dmem_oob, n_dmem_top_wr;
    reg [31:0] margin_wr_val, margin_rd_val;
    reg [31:0] sfr_score [0:3];
    integer n_sfr_score_rd;

    integer n_pcpi_err, n_crc_bad;

    // D5: FRAME double-buffer / bank monitors
    integer n_frame_wr, n_frame_wr_busy, n_frame_wr_active;
    integer n_l1_bank0, n_l1_bank1;

    // v5 weight path: how many direct l5_fc weight writes the blob produced (must be 256)
    integer n_l5_wr;

    // PC of the deliberate illegal funct3=7 instruction (the "CPU continued" check
    // must not depend on the firmware image size, which changed with D6).
    integer   ill_pc;

    integer n_tx;
    reg [7:0] txb [0:31];

    integer i;
    always @(posedge dut.clk) begin
        // ---- engine state census (frame length) ----
        if (dut.eng_state != 3'd0) n_st[dut.eng_state] = n_st[dut.eng_state] + 1;
    end

    always @(posedge dut.clk) begin
        // ---- PCPI ----
        if (dut.pcpi_valid && (dut.pcpi_insn[6:0] == 7'h0B)) begin
            if (dut.pcpi_ready) begin
                case (dut.pcpi_insn[14:12])
                    3'd0: n_conv = n_conv + 1;
                    3'd1: n_wait = n_wait + 1;
                    3'd2: begin
                              if (dut.pcpi_insn[31:25] <= 7'd3) begin
                                  acc_val[dut.pcpi_insn[31:25]] = dut.pcpi_rd;
                                  acc_got[dut.pcpi_insn[31:25]] = 1'b1;
                              end
                              n_acc = n_acc + 1;
                          end
                    3'd3, 3'd4, 3'd5, 3'd6: n_unknown_f3 = n_unknown_f3 + 1;
                    default: ;                       // 3'd7: the deliberate illegal one
                endcase
            end
            // the illegal instruction: must be acknowledged with wr=0 in the SAME cycle
            if (dut.pcpi_insn[14:12] == 3'd7) begin
                if (n_ill == 0) ill_pc = dut.u_cpu.reg_pc;
                n_ill = n_ill + 1;
                if (!dut.pcpi_ready || dut.pcpi_wr) n_ill_bad = n_ill_bad + 1;
            end
            // bnn.wait must stall the CPU (pcpi_wait) while the engine is running
            if ((dut.pcpi_insn[14:12] == 3'd1) && dut.pcpi_wait && !dut.pcpi_ready)
                n_wait_stall = n_wait_stall + 1;
        end

        // ---- CPU bus: SFR writes / reads and IMEM write leaks ----
        if (dut.mem_valid && dut.mem_ready) begin
            if (dut.mem_wstrb != 4'b0000) begin
                if (dut.mem_addr[31:16] == 16'h0001) begin
                    n_sfr_wr = n_sfr_wr + 1;
                    if (dut.mem_addr[5:0] == 6'h28) margin_wr_val = dut.mem_wdata;
                end
                if ((dut.mem_addr[31:16] == 16'h0000) && (dut.mem_addr[13] == 1'b0))
                    n_imem_wr = n_imem_wr + 1;
                if ((dut.mem_addr[31:16] == 16'h0000) && (dut.mem_addr[13] == 1'b1))
                    n_dmem_wr = n_dmem_wr + 1;
                // D6: any DMEM write in 0x3000..0x3FFF is outside the 4 KB window
                if ((dut.mem_addr[31:16] == 16'h0000) && (dut.mem_addr[13] == 1'b1) &&
                    (dut.mem_addr[12] == 1'b1))
                    n_dmem_oob = n_dmem_oob + 1;
                // D6: the top word 0x2FFC must be reachable through the CPU bus
                if ((dut.mem_addr[31:16] == 16'h0000) && (dut.mem_addr[13] == 1'b1) &&
                    (dut.mem_addr[12] == 1'b0) && (dut.mem_addr[11:0] == 12'hFFC))
                    n_dmem_top_wr = n_dmem_top_wr + 1;
            end else if (dut.mem_addr[31:16] == 16'h0001) begin
                case (dut.mem_addr[5:0])
                    6'h28: begin margin_rd_val = dut.mem_rdata; n_margin_rd = n_margin_rd + 1; end
                    6'h10: begin sfr_score[0] = dut.mem_rdata; n_sfr_score_rd = n_sfr_score_rd + 1; end
                    6'h14: begin sfr_score[1] = dut.mem_rdata; end
                    6'h18: begin sfr_score[2] = dut.mem_rdata; end
                    6'h30: begin sfr_score[3] = dut.mem_rdata; end
                    default: ;
                endcase
            end
        end

        // ---- UART result frame bytes ----
        if (dut.tx_start) begin
            if (n_tx < 32) txb[n_tx] = dut.tx_byte;
            n_tx = n_tx + 1;
        end

        // ---- D5: FRAME bank traffic ----
        if (dut.frame_we) begin
            n_frame_wr = n_frame_wr + 1;
            if (dut.eng_busy) begin
                n_frame_wr_busy = n_frame_wr_busy + 1;
                // a write that targets the bank the running inference reads would be
                // a double-buffer violation (and would corrupt the current result)
                if (dut.u_eng.frame_bank == dut.u_eng.frm_rd_bank_r)
                    n_frame_wr_active = n_frame_wr_active + 1;
            end
        end
        if (dut.eng_state == 3'd1) begin          // L1 = FRAME read phase
            if (dut.u_eng.frm_rd_bank_r) n_l1_bank1 = n_l1_bank1 + 1;
            else                         n_l1_bank0 = n_l1_bank0 + 1;
        end

        // ---- error strobes ----
        if (dut.pcpi_err)     n_pcpi_err = n_pcpi_err + 1;
        if (dut.frx_err_pulse) n_crc_bad = n_crc_bad + 1;

        // ---- v5 blob: one direct l5_fc weight write per FC beat ----
        if (dut.u_eng.u_blob.l5_we) n_l5_wr = n_l5_wr + 1;
    end

    // ================================================================ dut2 monitors (real v5 blob)
    integer n2_conv, n2_wait, n2_acc, n2_tx;
    reg [31:0] acc2_val [0:3];
    reg [7:0]  txb2 [0:31];

    always @(posedge dut2.clk) begin
        if (dut2.pcpi_valid && (dut2.pcpi_insn[6:0] == 7'h0B) && dut2.pcpi_ready) begin
            case (dut2.pcpi_insn[14:12])
                3'd0: n2_conv = n2_conv + 1;
                3'd1: n2_wait = n2_wait + 1;
                3'd2: begin
                          if (dut2.pcpi_insn[31:25] <= 7'd3)
                              acc2_val[dut2.pcpi_insn[31:25]] = dut2.pcpi_rd;
                          n2_acc = n2_acc + 1;
                      end
                default: ;
            endcase
        end
        if (dut2.tx_start) begin
            if (n2_tx < 32) txb2[n2_tx] = dut2.tx_byte;
            n2_tx = n2_tx + 1;
        end
    end

    // ================================================================ main sequence
    integer t0, err_before, imem0, crc_before;
    integer tmp;
    reg [7:0] rxcrc;

    integer stA [0:7];                    // engine state census snapshot after frame A
    integer accA [0:3], sfrA [0:3];       // frame A score snapshots
    integer waitA;
    integer split_byte;                   // D5: payload bytes sent while frame A runs
    reg [7:0] exp_s0, exp_s1, exp_s2;     // D2: expected saturating frame bytes

    // TB-side D2 reference: clamp(q >>> 10, -128, 127) on a 32 bit signed integer
    function [7:0] sat8_ref;
        input integer q;
        integer t;
        begin
            t = q >>> 10;                 // integer is signed -> arithmetic shift
            if      (t >  127) sat8_ref = 8'd127;
            else if (t < -128) sat8_ref = 8'h80;
            else               sat8_ref = t[7:0];
        end
    endfunction

    initial begin
        errors = 0;
        checks = 0;
        n_conv = 0; n_wait = 0; n_acc = 0; n_ill = 0; n_ill_bad = 0; n_unknown_f3 = 0;
        n_wait_stall = 0;
        n_sfr_wr = 0; n_imem_wr = 0; n_dmem_wr = 0; n_margin_rd = 0;
        n_dmem_oob = 0; n_dmem_top_wr = 0;
        n_tx = 0; n_sfr_score_rd = 0;
        n_pcpi_err = 0; n_crc_bad = 0;
        n_frame_wr = 0; n_frame_wr_busy = 0; n_frame_wr_active = 0;
        n_l1_bank0 = 0; n_l1_bank1 = 0;
        n_l5_wr = 0;
        n_wait_stall = 0; ill_pc = 0;
        n2_conv = 0; n2_wait = 0; n2_acc = 0; n2_tx = 0;
        for (i = 0; i < 4; i = i + 1) acc2_val[i] = 32'hxxxx_xxxx;
        for (i = 0; i < 32; i = i + 1) txb2[i] = 8'hxx;
        margin_wr_val = 32'hxxxx_xxxx;
        margin_rd_val = 32'hxxxx_xxxx;
        for (i = 0; i < 8; i = i + 1) n_st[i] = 0;
        for (i = 0; i < 4; i = i + 1) begin acc_val[i] = 32'hxxxx_xxxx; acc_got[i] = 1'b0; sfr_score[i] = 32'hxxxx_xxxx; end
        for (i = 0; i < 32; i = i + 1) txb[i] = 8'hxx;

        rst_n = 1'b0;
        rxd   = 1'b1;
        rxd2  = 1'b1;

        gen_stimulus;                          // frame A patterns + weights/theta
        build_blob;
        pack_payload;                          // payload = frame A
        build_ref;
        for (i = 0; i < 4; i = i + 1) begin
            refa_acc[i]   = ref_acc[i];
            refa_score[i] = ref_score[i];
        end
        refa_cls = exp_cls;
        for (i = 0; i < 512; i = i + 1) payload_a[i] = payload[i];

        // D5 second frame: bitwise inverse of A (a different 64x64 binary image).
        // Its 271 word weight blob is identical; only FRAME changes, so the two
        // inferences exercise the two FRAME banks with genuinely different data.
        for (i = 0; i < 4096; i = i + 1) frm[i] = ~frm_a[i];
        pack_payload;
        for (i = 0; i < 512; i = i + 1) payload_b[i] = payload[i];
        build_ref;
        for (i = 0; i < 4; i = i + 1) begin
            refb_acc[i]   = ref_acc[i];
            refb_score[i] = ref_score[i];
        end
        refb_cls = exp_cls;
        for (i = 0; i < 512; i = i + 1) payload[i] = payload_a[i];   // send A first

        $display("[tb] ---- stimulus ----");
        $display("[tb] W1=%018h", W1);
        $display("[tb] W2=%018h", W2);
        $display("[tb] W3=%016h", W3);
        $display("[tb] frame A ref acc   = %0d %0d %0d %0d",
                 refa_acc[0], refa_acc[1], refa_acc[2], refa_acc[3]);
        $display("[tb] frame A ref score = %0d %0d %0d %0d (Q6 19b signed)",
                 refa_score[0], refa_score[1], refa_score[2], refa_score[3]);
        $display("[tb] frame A ref class = %0d (margin gate 1536)", refa_cls);
        $display("[tb] frame B ref acc   = %0d %0d %0d %0d",
                 refb_acc[0], refb_acc[1], refb_acc[2], refb_acc[3]);
        $display("[tb] frame B ref score = %0d %0d %0d %0d (Q6 19b signed)",
                 refb_score[0], refb_score[1], refb_score[2], refb_score[3]);
        $display("[tb] frame B ref class = %0d (margin gate 1536)", refb_cls);

        // preload WBRAM before the reset is released (WB_INIT=0 -> no $readmemh race)
        for (i = 0; i < 271; i = i + 1) dut.wbmem[i] = blob[i];

        repeat (20) @(posedge clk100);
        imem0 = dut.imem[0];                  // snapshot before the CPU runs

        rst_n = 1'b1;

        // ---------------- 0. weight loader + CPU release ----------------
        // v5: 271 blob words + 36 theta push (the old class-major FC de-interleave pass
        // added 1,024 cycles; its removal is visible here and in the l5 write counter).
        t0 = 0;
        while ((dut.eng_ready !== 1'b1) && (t0 < 20000)) begin
            @(posedge clk100);
            t0 = t0 + 1;
        end
        $display("[tb] engine ready after %0d clk100 cycles (= %0d engine cycles; v5: 271 blob + 36 theta)",
                 t0, t0/2);
        chk("v5 weight load finished within 800 clk100 (400 engine cycles)", (t0 <= 800), 1);
        chk("v5 blob produced exactly 256 direct l5_fc weight writes", n_l5_wr, 256);

        t0 = 0;
        while ((dut.cpu_rst_n !== 1'b1) && (t0 < 200000)) begin
            @(posedge clk100);
            t0 = t0 + 1;
        end
        chk("engine weights loaded, CPU released from reset", (dut.cpu_rst_n === 1'b1), 1);
        chk32("bnn_engine.score_o == 0 before the first frame", dut.eng_score[18:0], 19'd0);

        // ---------------- 1. firmware boot + DMEM 4 KB map + SFR write/read ----------
        t0 = 0;
        while ((n_margin_rd == 0) && (t0 < 200000)) begin
            @(posedge clk100);
            t0 = t0 + 1;
        end
        chk("firmware wrote SFR MARGIN (CPU alive)", (n_sfr_wr > 0), 1);
        chk32("SFR MARGIN write value", margin_wr_val, 32'h0000_A55A);
        chk32("SFR MARGIN read back (write/read consistent)", margin_rd_val, 32'h0000_A55A);
        chk("no write ever reached IMEM (SFR does not leak)", n_imem_wr, 0);
        chk("firmware used the stack in DMEM", (n_dmem_wr > 0), 1);
        chk32("IMEM word 0 unchanged", dut.imem[0], imem0);

        // ---------------- 1a. D6: DMEM 4 KB map (firmware writes / TB reads back) ----
        chk32("DMEM top word 0x2FFC read back", dut.dmem[1023], 32'h5A5A_2FFC);
        chk32("DMEM bottom word 0x2000 read back", dut.dmem[0], 32'hC3C3_2000);
        chk32("DMEM self-test pass word @0x2004", dut.dmem[1], 32'h600D_BEEF);
        chk("no CPU write outside DMEM 0x2000..0x2FFF", n_dmem_oob, 0);
        chk("CPU reached the DMEM top word 0x2FFC", (n_dmem_top_wr > 0), 1);

        // ---------------- 2. illegal PCPI funct3=7 must not hang ----------------
        t0 = 0;
        while ((n_ill == 0) && (t0 < 100000)) begin
            @(posedge clk100);
            t0 = t0 + 1;
        end
        chk("illegal Custom-0 funct3=7 executed", n_ill, 1);
        chk("illegal funct3=7 answered with ready=1 / wr=0 (no hang)", n_ill_bad, 0);
        // give the CPU a few cycles to step past the illegal instruction before the
        // PC-progress check (the monitor may sample reg_pc on the ack cycle itself)
        repeat (20) @(posedge clk100);
        chk("CPU continued after the illegal instruction", dut.u_cpu.reg_pc > ill_pc, 1);

        // ---------------- 3. corrupted CRC frame must be rejected ----------------
        repeat (40) @(posedge dut.clk);         // let the error counter settle
        err_before = dut.u_sfr.err_cnt;
        crc_before = n_crc_bad;
        send_frame(1);
        repeat (400) @(posedge dut.clk);
        chk("bad CRC frame did not raise frame_ready", dut.u_sfr.st_frame_rdy, 0);
        chk("bad CRC frame raised exactly one crc_err strobe", n_crc_bad - crc_before, 1);
        chk("bad CRC frame incremented ERRORS by 1", dut.u_sfr.err_cnt - err_before, 1);
        chk("STATUS.crc_err sticky", dut.u_sfr.st_crc, 1);
        chk("engine did not start on the bad frame", n_conv, 0);

        // ---------------- 4. frame A -> full inference ----------------
        send_frame(0);                          // payload = A
        t0 = 0;
        while ((n_conv == 0) && (t0 < 200000)) begin
            @(posedge clk100);
            t0 = t0 + 1;
        end
        chk("frame A started (bnn.conv seen)", (n_conv > 0), 1);

        // D5 check (1): stream the first 200 payload bytes of frame B *while* frame A
        // is being computed.  The capture path writes ~cap_bank (bank 0) while the
        // engine reads bank 1, so frame A's result must not change.  At 160 clk100
        // per byte this takes 16,320 clk < 16,911 clk of one inference.
        split_byte = 200;
        for (i = 0; i < 512; i = i + 1) payload[i] = payload_b[i];
        send_frame_prefix(split_byte);

        t0 = 0;
        while ((n_tx < 10) && (t0 < 800000)) begin
            @(posedge clk100);
            t0 = t0 + 1;
        end
        chk("10 B result frame transmitted", n_tx, 10);
        chk("bnn.conv issued exactly once", n_conv, 1);
        chk("bnn.wait completed", n_wait, 1);
        $display("[tb] frame A bnn.wait held the CPU for %0d cycles (pcpi_wait)", n_wait_stall);
        chk("bnn.wait stalled the CPU while the engine ran (pcpi_wait)",
            (n_wait_stall > 16800), 1);
        chk("bnn.acc issued 4 times", n_acc, 4);

        // ---------------- 5. frame A: scores, bytes, bank behaviour -----------------
        for (i = 0; i < 4; i = i + 1) begin
            accA[i] = acc_val[i];
            sfrA[i] = sfr_score[i];
        end
        waitA = n_wait_stall;

        for (i = 0; i < 4; i = i + 1) begin
            case (i)
                0: begin
                    chk32("frame A bnn.acc(0) == reference", accA[0], refa_score[0]);
                    chk32("frame A SFR SCORE0 == reference", sfrA[0], refa_score[0]);
                end
                1: begin
                    chk32("frame A bnn.acc(1) == reference", accA[1], refa_score[1]);
                    chk32("frame A SFR SCORE1 == reference", sfrA[1], refa_score[1]);
                end
                2: begin
                    chk32("frame A bnn.acc(2) == reference", accA[2], refa_score[2]);
                    chk32("frame A SFR SCORE2 == reference", sfrA[2], refa_score[2]);
                end
                default: begin
                    chk32("frame A bnn.acc(3) == reference", accA[3], refa_score[3]);
                    chk32("frame A SFR SCORE3 == reference", sfrA[3], refa_score[3]);
                end
            endcase
        end
        chk32("frame A engine acc_o class 0 == ref", {20'd0, dut.eng_acc[0*12 +: 12]}, refa_acc[0]);
        chk32("frame A engine acc_o class 3 == ref", {20'd0, dut.eng_acc[3*12 +: 12]}, refa_acc[3]);

        exp_s0 = sat8_ref(refa_score[0]);
        exp_s1 = sat8_ref(refa_score[1]);
        exp_s2 = sat8_ref(refa_score[2]);
        $display("[tb] frame A Q6 %0d/%0d/%0d -> sat8 %0d/%0d/%0d",
                 refa_score[0], refa_score[1], refa_score[2], exp_s0, exp_s1, exp_s2);

        chk("frame A byte0 = 0xA5", txb[0], 8'hA5);
        chk("frame A byte1 = 0x83", txb[1], 8'h83);
        chk("frame A byte2 = class", txb[2], refa_cls);
        chk("frame A byte3 = sat8(score0 >>> 10)", txb[3], exp_s0);
        chk("frame A byte4 = sat8(score1 >>> 10)", txb[4], exp_s1);
        chk("frame A byte5 = sat8(score2 >>> 10)", txb[5], exp_s2);
        chk("frame A byte6 bit0 = done", txb[6] & 8'h01, 8'h01);
        chk("frame A byte6 bit1 = busy (must be 0)", txb[6] & 8'h02, 8'h00);
        chk("frame A byte6 bit2 = crc_err (from rejected)", txb[6] & 8'h04, 8'h04);
        chk("frame A byte7 = fps (frame counter)", txb[7], 8'd1);
        chk("frame A byte8 = frame seq", txb[8], 8'd1);
        rxcrc = 8'h00;
        for (i = 0; i < 9; i = i + 1) rxcrc = crc8(rxcrc, txb[i]);
        chk("frame A byte9 = CRC8 of bytes 0..8", txb[9], rxcrc);

        // ---------------- 6. D5: the mid-frame frame B went to the spare bank ------
        $display("[tb] FRAME writes: total=%0d, during frame A=%0d, active-bank hits=%0d",
                 n_frame_wr, n_frame_wr_busy, n_frame_wr_active);
        chk("frame B bytes written while frame A computing", (n_frame_wr_busy > 0), 1);
        chk("no FRAME write hit the bank frame A was reading", n_frame_wr_active, 0);
        chk("frame A L1 read bank 1 (the published frame)", (n_l1_bank1 > 0), 1);
        chk("frame A L1 never read the half-written bank 0", n_l1_bank0, 0);

        // ---------------- 7. frame A: length = 16,911 cycles ------------------------
        for (i = 0; i < 8; i = i + 1) stA[i] = n_st[i];
        chk("frame A S_L1 cycles", stA[1], EXP_L1);
        chk("frame A S_L1_DONE barrier cycles", stA[2], EXP_BAR);
        chk("frame A S_L2 cycles", stA[3], EXP_L2);
        chk("frame A S_L3 cycles", stA[4], EXP_L3);
        chk("frame A S_L4 cycles", stA[5], EXP_L4);
        chk("frame A S_L5 cycles", stA[6], EXP_L5);
        chk("frame A S_DONE cycles", stA[7], EXP_DONE);
        chk("frame A total = 16,911 cycles",
            stA[1]+stA[2]+stA[3]+stA[4]+stA[5]+stA[6]+stA[7], EXP_FRAME);

        // ---------------- 8. frame B: second inference out of bank 0 ----------------
        send_frame_suffix(split_byte);
        t0 = 0;
        while (((n_conv < 2) || (n_tx < 20)) && (t0 < 1000000)) begin
            @(posedge clk100);
            t0 = t0 + 1;
        end
        chk("frame B received and started (bnn.conv #2)", n_conv, 2);
        chk("20 B result bytes transmitted (2 frames)", n_tx, 20);
        chk("bnn.wait completed twice", n_wait, 2);
        $display("[tb] frame B bnn.wait held the CPU for %0d cycles", n_wait_stall - waitA);
        chk("frame B also stalled the CPU ~16,900 cycles", ((n_wait_stall - waitA) > 16800), 1);
        chk("bnn.acc issued 8 times (4 per frame)", n_acc, 8);

        for (i = 0; i < 4; i = i + 1) begin
            case (i)
                0: begin
                    chk32("frame B bnn.acc(0) == reference", acc_val[0], refb_score[0]);
                    chk32("frame B SFR SCORE0 == reference", sfr_score[0], refb_score[0]);
                end
                1: begin
                    chk32("frame B bnn.acc(1) == reference", acc_val[1], refb_score[1]);
                    chk32("frame B SFR SCORE1 == reference", sfr_score[1], refb_score[1]);
                end
                2: begin
                    chk32("frame B bnn.acc(2) == reference", acc_val[2], refb_score[2]);
                    chk32("frame B SFR SCORE2 == reference", sfr_score[2], refb_score[2]);
                end
                default: begin
                    chk32("frame B bnn.acc(3) == reference", acc_val[3], refb_score[3]);
                    chk32("frame B SFR SCORE3 == reference", sfr_score[3], refb_score[3]);
                end
            endcase
        end
        chk32("frame B engine acc_o class 0 == ref", {20'd0, dut.eng_acc[0*12 +: 12]}, refb_acc[0]);
        chk32("frame B engine acc_o class 3 == ref", {20'd0, dut.eng_acc[3*12 +: 12]}, refb_acc[3]);

        exp_s0 = sat8_ref(refb_score[0]);
        exp_s1 = sat8_ref(refb_score[1]);
        exp_s2 = sat8_ref(refb_score[2]);
        $display("[tb] frame B Q6 %0d/%0d/%0d -> sat8 %0d/%0d/%0d",
                 refb_score[0], refb_score[1], refb_score[2], exp_s0, exp_s1, exp_s2);

        chk("frame B byte0 = 0xA5", txb[10], 8'hA5);
        chk("frame B byte1 = 0x83", txb[11], 8'h83);
        chk("frame B byte2 = class", txb[12], refb_cls);
        chk("frame B byte3 = sat8(score0 >>> 10)", txb[13], exp_s0);
        chk("frame B byte4 = sat8(score1 >>> 10)", txb[14], exp_s1);
        chk("frame B byte5 = sat8(score2 >>> 10)", txb[15], exp_s2);
        chk("frame B byte6 bit0 = done", txb[16] & 8'h01, 8'h01);
        chk("frame B byte6 bit1 = busy (must be 0)", txb[16] & 8'h02, 8'h00);
        chk("frame B byte7 = fps (frame counter)", txb[17], 8'd2);
        chk("frame B byte8 = frame seq", txb[18], 8'd2);
        rxcrc = 8'h00;
        for (i = 10; i < 19; i = i + 1) rxcrc = crc8(rxcrc, txb[i]);
        chk("frame B byte9 = CRC8 of bytes 10..18", txb[19], rxcrc);

        chk("frame B S_L1 cycles", n_st[1]-stA[1], EXP_L1);
        chk("frame B S_L1_DONE barrier cycles", n_st[2]-stA[2], EXP_BAR);
        chk("frame B S_L2 cycles", n_st[3]-stA[3], EXP_L2);
        chk("frame B S_L3 cycles", n_st[4]-stA[4], EXP_L3);
        chk("frame B S_L4 cycles", n_st[5]-stA[5], EXP_L4);
        chk("frame B S_L5 cycles", n_st[6]-stA[6], EXP_L5);
        chk("frame B S_DONE cycles", n_st[7]-stA[7], EXP_DONE);
        chk("frame B total = 16,911 cycles",
            (n_st[1]-stA[1])+(n_st[2]-stA[2])+(n_st[3]-stA[3])+(n_st[4]-stA[4])+
            (n_st[5]-stA[5])+(n_st[6]-stA[6])+(n_st[7]-stA[7]), EXP_FRAME);

        chk("frame B L1 read bank 0 (the new active bank)", (n_l1_bank0 > 0), 1);
        chk("active FRAME bank is 0 after frame B", dut.cap_bank, 0);
        chk("frame B reference really differs from A",
            (refb_score[0] != refa_score[0]) || (refb_score[1] != refa_score[1]) ||
            (refb_score[2] != refa_score[2]) || (refb_score[3] != refa_score[3]), 1);

        // ---------------- 9. misc ----------------
        chk("engine idle at the end", dut.eng_busy, 0);
        chk("frame counter == 2", dut.u_sfr.frame_cnt, 2);
        chk("frame_rx seq == 2 (two good frames)", dut.frx_seq, 2);
        chk("frame_rx n_ok == 2", dut.frx_nok, 2);
        chk("no unknown-funct3 Custom-0 instruction", n_unknown_f3, 0);
        chk("ERRORS == pcpi_err + crc_err strobes", dut.u_sfr.err_cnt, n_pcpi_err + n_crc_bad);

        // ---------------- 10. D2: directed saturating scaler checks -----------------
        // Force the engine score register after the last frame; cur_s0..s2 are the
        // same combinational sat8() path that feeds the result-frame snapshot.
        force dut.u_eng.score_r = {-19'sd131073, 19'sd131071, -19'sd131072, 19'sd131072};
        #1;
        chk("D2 +131072 >>> 10 -> +127 (saturate)", dut.cur_s0, 127);
        chk("D2 -131072 >>> 10 -> -128 (no clamp)", $signed(dut.cur_s1), -128);
        chk("D2 +131071 >>> 10 -> +127", dut.cur_s2, 127);
        force dut.u_eng.score_r = {19'sd0, -19'sd65536, 19'sd65536, -19'sd131073};
        #1;
        chk("D2 -131073 >>> 10 -> -128 (saturate)", $signed(dut.cur_s0), -128);
        chk("D2 +65536 >>> 10 -> +64", dut.cur_s1, 64);
        chk("D2 -65536 >>> 10 -> -64", $signed(dut.cur_s2), -64);
        force dut.u_eng.score_r = {19'sd0, 19'sd0, -19'sd1024, 19'sd1023};
        #1;
        chk("D2 +1023 >>> 10 -> 0", dut.cur_s0, 0);
        chk("D2 -1024 >>> 10 -> -1", $signed(dut.cur_s1), -1);
        release dut.u_eng.score_r;

        // ---------------- 11. real v5 weights.hex -> dut2 -> 4 scores ---------------
        // dut2 boots from the real files (WB_INIT=1 -> $readmemh weights.hex).  The TB
        // parses the same 271 word blob with its own decoder and runs the same Verilog
        // array reference, so this passes only if gen_weights' beat-major FC order, the
        // RTL live stream into l5_fc and the theta unpack all agree bit for bit.
        parse_real_blob;

        t0 = 0;
        while ((dut2.cpu_rst_n !== 1'b1) && (t0 < 200000)) begin
            @(posedge clk100);
            t0 = t0 + 1;
        end
        chk("dut2 (real v5 blob): weights loaded, CPU released", (dut2.cpu_rst_n === 1'b1), 1);
        chk("dut2 (real v5 blob): score_o == 0 before the first frame",
            dut2.eng_score[18:0], 19'd0);

        // swap the real, file-parsed weights/theta into the shared reference arrays
        W1 = W1R; W2 = W2R; W3 = W3R;
        for (i = 0; i < 4; i = i + 1) WFA[i] = WFA_R[i];
        for (i = 0; i < 8; i = i + 1) begin
            TH1[i] = TH1R[i]; TH2[i] = TH2R[i]; TH3[i] = TH3R[i]; TH4[i] = TH4R[i];
        end
        for (i = 0; i < 4; i = i + 1) THFC[i] = THFCR[i];

        // frame C: a third deterministic 64x64 image (different from A and B)
        for (i = 0; i < 4096; i = i + 1) begin
            tmp = 0;
            if (((i % 64) >= 8) && ((i % 64) <= 55) && (((i/64) % 7) < 4)) tmp = 1;
            if ((((i % 64)*5 + (i/64)*3) % 17) == 0) tmp = 1;
            frm[i] = (tmp != 0);
        end
        pack_payload;
        build_ref;
        for (i = 0; i < 4; i = i + 1) begin
            refc_acc[i]   = ref_acc[i];
            refc_score[i] = ref_score[i];
        end
        refc_cls = exp_cls;
        $display("[tb] frame C (real v5 weights) ref acc   = %0d %0d %0d %0d",
                 refc_acc[0], refc_acc[1], refc_acc[2], refc_acc[3]);
        $display("[tb] frame C (real v5 weights) ref score = %0d %0d %0d %0d",
                 refc_score[0], refc_score[1], refc_score[2], refc_score[3]);
        $display("[tb] frame C (real v5 weights) ref class = %0d", refc_cls);

        send_frame2(0);
        t0 = 0;
        while (((n2_conv == 0) || (n2_tx < 10)) && (t0 < 1200000)) begin
            @(posedge clk100);
            t0 = t0 + 1;
        end
        chk("dut2 (real v5 blob): frame C started (bnn.conv)", (n2_conv > 0), 1);
        chk("dut2 (real v5 blob): 10 B result frame transmitted", n2_tx, 10);
        chk("dut2 (real v5 blob): bnn.wait completed", n2_wait, 1);

        // 4 scores through the instruction path, the SFR path and the raw acc_o port
        chk32("frame C bnn.acc(0) == TB real-blob reference", acc2_val[0], refc_score[0]);
        chk32("frame C bnn.acc(1) == TB real-blob reference", acc2_val[1], refc_score[1]);
        chk32("frame C bnn.acc(2) == TB real-blob reference", acc2_val[2], refc_score[2]);
        chk32("frame C bnn.acc(3) == TB real-blob reference", acc2_val[3], refc_score[3]);
        chk32("frame C SFR SCORE0 == TB real-blob reference",
              {{13{dut2.u_sfr.sc0[18]}}, dut2.u_sfr.sc0}, refc_score[0]);
        chk32("frame C SFR SCORE1 == TB real-blob reference",
              {{13{dut2.u_sfr.sc1[18]}}, dut2.u_sfr.sc1}, refc_score[1]);
        chk32("frame C SFR SCORE2 == TB real-blob reference",
              {{13{dut2.u_sfr.sc2[18]}}, dut2.u_sfr.sc2}, refc_score[2]);
        chk32("frame C SFR SCORE3 == TB real-blob reference",
              {{13{dut2.u_sfr.sc3[18]}}, dut2.u_sfr.sc3}, refc_score[3]);
        chk32("frame C engine acc_o class 0 == ref",
              {20'd0, dut2.eng_acc[0*12 +: 12]}, refc_acc[0]);
        chk32("frame C engine acc_o class 1 == ref",
              {20'd0, dut2.eng_acc[1*12 +: 12]}, refc_acc[1]);
        chk32("frame C engine acc_o class 2 == ref",
              {20'd0, dut2.eng_acc[2*12 +: 12]}, refc_acc[2]);
        chk32("frame C engine acc_o class 3 == ref",
              {20'd0, dut2.eng_acc[3*12 +: 12]}, refc_acc[3]);

        // result frame bytes (same D2 saturating scaler as frame A/B)
        exp_s0 = sat8_ref(refc_score[0]);
        exp_s1 = sat8_ref(refc_score[1]);
        exp_s2 = sat8_ref(refc_score[2]);
        chk("frame C byte0 = 0xA5", txb2[0], 8'hA5);
        chk("frame C byte1 = 0x83", txb2[1], 8'h83);
        chk("frame C byte2 = class", txb2[2], refc_cls);
        chk("frame C byte3 = sat8(score0 >>> 10)", txb2[3], exp_s0);
        chk("frame C byte4 = sat8(score1 >>> 10)", txb2[4], exp_s1);
        chk("frame C byte5 = sat8(score2 >>> 10)", txb2[5], exp_s2);
        chk("frame C byte6 bit0 = done", txb2[6] & 8'h01, 8'h01);
        chk("frame C byte6 bit1 = busy (must be 0)", txb2[6] & 8'h02, 8'h00);
        chk("frame C byte7 = fps (dut2 frame counter)", txb2[7], 8'd1);
        chk("frame C byte8 = frame seq (dut2)", txb2[8], 8'd1);
        rxcrc = 8'h00;
        for (i = 0; i < 9; i = i + 1) rxcrc = crc8(rxcrc, txb2[i]);
        chk("frame C byte9 = CRC8 of bytes 0..8", txb2[9], rxcrc);

        chk("dut2 (real v5 blob): engine idle after frame C", dut2.eng_busy, 0);


        $display("");
        $display("=========================================================");
        $display("[tb] checks = %0d, failures = %0d", checks, errors);
        if (errors == 0) $display("[PASS] tb_top_bnn: ALL %0d CHECKS PASSED", checks);
        else             $display("[FAIL] tb_top_bnn: %0d of %0d checks FAILED", errors, checks+errors);
        $display("=========================================================");
        $finish;
    end

    // ---------------- watchdog ----------------
    initial begin
        repeat (3000000) @(posedge clk100);
        $display("[FAIL] tb_top_bnn: global timeout (3,000,000 clk100 cycles)");
        $display("[tb] checks = %0d, failures = %0d", checks, errors + 1);
        $display("[FAIL] tb_top_bnn: TIMEOUT");
        $finish;
    end

endmodule
