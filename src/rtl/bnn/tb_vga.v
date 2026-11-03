// ============================================================================
// tb_vga.v -- T5 VGA self-check for vga_simple (640x480@60, 12-bit 4:4:4)
// ----------------------------------------------------------------------------
// The DUT is stimulated with exactly the signals top_bnn provides (class, four
// Q6 scores, busy/done/err) -- there is NO CPU and NO UART in this test, which is
// what the T5 acceptance criterion asks for ("pull the serial cable and it still
// displays").
//
// Checked (every pixel is sampled from the real rgb output):
//   * raster: HS/VS period, sync width, frame rate 59.5 +-0.5 Hz
//   * pixel tick: rgb advances only on the 25 MHz tick and stays locked to the
//     raster counter; rgb == 0 in both blanking intervals
//   * title colour bars (8 x 80 px, exact colours, boundary x=79/80)
//   * class name for class 0..3: ink/background pixels, text colour, ink-pixel
//     count and bounding box.  The expected numbers come from font_spec_check.py,
//     i.e. from the ASCII art of the font, NOT from the RTL.
//   * score bars: len = clamp(score>>>6 + 256, 0, 512) for two different score
//     vectors, per-bar monotonicity and saturation at both ends
//   * status word: IDLE / BUSY / DONE (latched from a 1-cycle pulse) / ERR,
//     BUSY priority over ERR, and BUSY clearing a latched DONE
//   * T5: the very first frame after reset, with no stimulus at all, is already a
//     complete self-test picture (colour bars + class name + IDLE + four bars)
//
// Frame capture is white-box for the *coordinate* only (dut.hcnt/dut.vcnt); the
// data is the real vga_rgb output.  vga_simple registers rgb, hs and vs together
// from the same next-state counter, so (hcnt, vcnt, vga_rgb) always describe the
// same pixel: the capture below stores each pixel exactly once per frame.
//
// Negative control: `xvlog -d NEGCTL` makes one title-bar expectation wrong on
// purpose; the run must then report at least one FAIL (proves the TB has teeth).
//
// Verilog-2001.  Run: see 04_娴嬭瘯/20261101_VGA浠跨湡/run_vga.ps1
// ============================================================================
`timescale 1ns / 1ps

module tb_vga;

    // ---------------------------------------------------------------- timing
    localparam integer CLK_NS  = 20;                   // 50 MHz system clock
    localparam integer PIX_CK  = 2;                    // clocks per pixel (25 MHz)
    localparam integer HS_CLK  = 800 * PIX_CK;         // 1600 clk = 32.000 us
    localparam integer HS_LOW  = 96  * PIX_CK;         // 192  clk = 3.840 us
    localparam integer HS_HIGH = 704 * PIX_CK;         // 1408 clk = 28.160 us
    localparam integer VS_CLK  = 525 * 800 * PIX_CK;   // 840000 clk = 16.800 ms
    localparam integer VS_LOW  = 2 * 800 * PIX_CK;     // 3200 clk = 64.000 us
    localparam real    FPS     = 50000000.0 / 840000.0;// 59.5238 Hz

    // ---------------------------------------------------------------- layout
    localparam integer TITLE_H = 40;
    localparam integer SEP_Y   = 42;
    localparam integer TXT_Y   = 96;
    localparam integer BAR_Y   = 200;
    localparam integer BAR_P   = 36;
    localparam integer BAR_X   = 80;
    localparam integer BAR_MAX = 512;
    localparam integer STA_Y   = 432;
    localparam integer STA_X   = 16;

    // ---------------------------------------------------------------- colours
    localparam [11:0] C_BG  = 12'h024;
    localparam [11:0] C_SEP = 12'h444;
    localparam [11:0] C_TRK = 12'h222;

`ifdef NEGCTL
    localparam [11:0] C_BAR0_EXP = 12'h0F0;   // negative control: deliberately wrong
`else
    localparam [11:0] C_BAR0_EXP = 12'hFFF;
`endif

    // ---------------------------------------------------------------- DUT
    reg         clk    = 1'b0;
    reg         rst_n  = 1'b0;
    reg         pix_en = 1'b1;
    reg  [2:0]  cls    = 3'd0;
    reg  [75:0] score  = 76'd0;
    reg         busy   = 1'b0;
    reg         done   = 1'b0;
    reg         err    = 1'b0;
    wire        vga_hs, vga_vs;
    wire [11:0] vga_rgb;

    vga_simple dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .pix_clk_en (pix_en),
        .cls        (cls),
        .score      (score),
        .busy       (busy),
        .done       (done),
        .err        (err),
        .vga_hs     (vga_hs),
        .vga_vs     (vga_vs),
        .vga_rgb    (vga_rgb)
    );

    always #(CLK_NS/2) clk = ~clk;                 // 50 MHz

    // ---------------------------------------------------------------- monitors
    // clock cycle counter, updated on the falling edge so it never races with an
    // hs/vs edge (both change on the rising edge)
    integer cyc = 0;
    always @(negedge clk) cyc = cyc + 1;

    integer hs_edges = 0;
    always @(posedge vga_hs) hs_edges = hs_edges + 1;

    // blanking must be black for the whole run
    integer hb_err = 0, vb_err = 0;
    always @(posedge clk) begin
        if (rst_n === 1'b1) begin
            if ((dut.hcnt >= 640) && (vga_rgb !== 12'h000)) hb_err = hb_err + 1;
            if ((dut.vcnt >= 480) && (vga_rgb !== 12'h000)) vb_err = vb_err + 1;
        end
    end

    // frame capture: one entry per active pixel of the last complete frame
    reg [11:0] frm [0:307199];
    always @(posedge clk) begin
        if ((dut.hcnt < 640) && (dut.vcnt < 480))
            frm[dut.vcnt*640 + dut.hcnt] <= vga_rgb;
    end

    // ---------------------------------------------------------------- helpers
    integer npass = 0;
    integer nfail = 0;

    task chk;
        input [8*76-1:0] name;
        input            cond;
        begin
            if (cond) begin
                npass = npass + 1;
                $display("[PASS] %0s", name);
            end else begin
                nfail = nfail + 1;
                $display("[FAIL] %0s", name);
            end
        end
    endtask

    task get_px;
        input  integer xx;
        input  integer yy;
        output [11:0]  c;
        begin
            c = frm[yy*640 + xx];
        end
    endtask

    // wait until one complete frame has been captured with the current inputs
    task wait_frame;
        begin
            @(posedge vga_vs);
            @(posedge vga_vs);
        end
    endtask

    // expected bar length for a raw 19-bit Q6 score (independent re-derivation)
    function integer explen;
        input integer q;
        integer v;
        begin
            v = (q >>> 6) + 256;
            if (v < 0)   v = 0;
            if (v > 512) v = 512;
            explen = v;
        end
    endfunction

    task set_score;
        input integer q0, q1, q2, q3;
        begin
            score[18:0]  = q0;      // 19-bit two's complement truncation
            score[37:19] = q1;
            score[56:38] = q2;
            score[75:57] = q3;
        end
    endtask

    // ink statistics of a 32-pixel-high text band (whole width)
    task measure_band;
        input  integer y0;
        input  [11:0]  ink_col;
        output integer lx;
        output integer rx;
        output integer cnt;
        output [63:0]  sig;
        integer xx, yy;
        reg [11:0] c;
        begin
            lx = 65535; rx = -1; cnt = 0; sig = 64'd0;
            for (yy = y0; yy < y0 + 32; yy = yy + 1) begin
                for (xx = 0; xx < 640; xx = xx + 1) begin
                    c = frm[yy*640 + xx];
                    if (c === ink_col) begin
                        cnt = cnt + 1;
                        if (xx < lx) lx = xx;
                        if (xx > rx) rx = xx;
                        sig = (sig * 64'd33) + {32'd0, xx[9:0], yy[9:0]};
                    end
                end
            end
        end
    endtask

    // length of score bar bi (contiguous run of its colour from x = BAR_X)
    task measure_bar;
        input  integer bi;
        input  [11:0]  col;
        output integer len;
        integer xx;
        reg [11:0] c;
        begin
            len = 0;
            xx  = BAR_X;
            while (xx < BAR_X + BAR_MAX) begin
                c = frm[(BAR_Y + bi*BAR_P + 12)*640 + xx];
                if (c === col) begin
                    len = len + 1;
                    xx  = xx + 1;
                end else begin
                    xx = BAR_X + BAR_MAX;          // stop at the first gap
                end
            end
        end
    endtask

    // number of active pixels that are not the background colour
    // (a task, not a function: Verilog-2001 functions need >= 1 input)
    task frame_nonbg;
        output integer n;
        integer ii;
        begin
            n = 0;
            for (ii = 0; ii < 307200; ii = ii + 1)
                if (frm[ii] !== C_BG) n = n + 1;
        end
    endtask

    // dump the captured frame as binary PPM (P6) for eyeballing
    task dump_ppm;
        input [8*40-1:0] fname;
        integer fd, ii;
        reg [11:0] c;
        reg [7:0] r8, g8, b8;
        begin
            fd = $fopen(fname, "wb");
            if (fd == 0) begin
                $display("[WARN] cannot open %0s", fname);
            end else begin
                $fwrite(fd, "P6\n640 480\n255\n");
                for (ii = 0; ii < 307200; ii = ii + 1) begin
                    c  = frm[ii];
                    r8 = {c[11:8], c[11:8]};
                    g8 = {c[7:4],  c[7:4]};
                    b8 = {c[3:0],  c[3:0]};
                    $fwrite(fd, "%c%c%c", r8, g8, b8);
                end
                $fclose(fd);
                $display("[tb] frame dumped to %0s", fname);
            end
        end
    endtask

    // ---------------------------------------------------------------- stimulus
    integer i, k;
    integer t0, t1, hs_p, hs_lo, hs_hi, vs_p, vs_lo;
    integer lx, rx, cnt, ok, viol, len;
    integer len_a [0:3];
    integer len_b [0:3];
    integer cnt_cls [0:3];
    reg [63:0] sig_cls [0:3];
    reg [63:0] sig_idle, sig_busy, sig_done, sig_err;
    reg [11:0] cp, cq;
    reg [9:0]  h_prev;
    reg [11:0] r_prev;
    real       f_hz;
    integer    nonbg;

    initial begin
        $display("=========================================================");
        $display("[tb_vga] T5 VGA self-check -- 640x480@60, no CPU, no UART");
        $display("=========================================================");

        // ================= phase 0: power-on, NO stimulus at all =============
        // Exactly what the board does with the serial cable unplugged: the DUT
        // sees reset, then idle inputs forever.
        #200 rst_n = 1'b1;
        wait_frame;                              // first complete frame

        frame_nonbg(nonbg);
        $display("[tb] idle frame: %0d non-background pixels", nonbg);
        chk("T5 no-CPU self-test: first frame is not blank (>=25000 px)",
            nonbg >= 25000);

        get_px(320, 10, cp);
        chk("T5 no-CPU self-test: title colour bar drawn at (320,10)",
            cp === 12'hF0F);

        get_px(260, 97, cp);
        chk("T5 no-CPU self-test: class name ink at (260,97) [ROCK]", cp === 12'hF00);

        get_px(84, 433, cp);
        chk("T5 no-CPU self-test: status word ink at (84,433) [IDLE]", cp === 12'h888);

        measure_bar(0, 12'hF00, len);
        measure_bar(1, 12'h0F0, len_b[1]);
        measure_bar(2, 12'h0FF, len_b[2]);
        measure_bar(3, 12'hFA0, len_b[3]);
        $display("[tb] idle bar lengths = %0d %0d %0d %0d (score 0 -> 256 each)",
                 len, len_b[1], len_b[2], len_b[3]);
        chk("T5 no-CPU self-test: 4 score bars drawn, score 0 -> 256 px",
            (len == 256) && (len_b[1] == 256) && (len_b[2] == 256) && (len_b[3] == 256));

        dump_ppm("vga_frame_idle.ppm");             // T5 evidence: the bare-board frame

        // ================= phase 1: raster timing ============================
        @(posedge vga_hs); t0 = cyc;
        @(posedge vga_hs); t1 = cyc;
        hs_p = t1 - t0;
        $display("[tb] hsync period = %0d clk = %0d.%03d us", hs_p, hs_p*20/1000,
                 hs_p*20 % 1000);
        chk("hsync period = 1600 clk (32.000 us, 800 px @25MHz)", hs_p == HS_CLK);

        @(negedge vga_hs); t0 = cyc;
        @(posedge vga_hs); t1 = cyc;
        hs_lo = t1 - t0;
        @(posedge vga_hs); t0 = cyc;
        @(negedge vga_hs); t1 = cyc;
        hs_hi = t1 - t0;
        $display("[tb] hsync low = %0d clk (3.840 us), high = %0d clk (28.160 us)",
                 hs_lo, hs_hi);
        chk("hsync low = 192 clk (3.840 us, 96 px)",  hs_lo == HS_LOW);
        chk("hsync high = 1408 clk (28.160 us, 704 px)", hs_hi == HS_HIGH);

        @(posedge vga_vs); t0 = cyc;
        @(posedge vga_vs); t1 = cyc;
        vs_p = t1 - t0;
        $display("[tb] vsync period = %0d clk = %0d us", vs_p, vs_p*20/1000);
        chk("vsync period = 840000 clk (16.800 ms, 525 lines)", vs_p == VS_CLK);

        @(negedge vga_vs); t0 = cyc;
        @(posedge vga_vs); t1 = cyc;
        vs_lo = t1 - t0;
        $display("[tb] vsync low = %0d clk = %0d us", vs_lo, vs_lo*20/1000);
        chk("vsync low = 3200 clk (64.000 us, 2 lines)", vs_lo == VS_LOW);

        f_hz = 50000000.0 / (vs_p * 1.0);
        $display("[tb] frame rate = %0.4f Hz", f_hz);
        chk("frame rate 59.5 +-0.5 Hz (measured 59.52)", (f_hz > 59.0) && (f_hz < 60.0));

        // ================= phase 2: pixel tick / blanking ====================
        ok = 0; viol = 0;
        for (i = 0; i < 256; i = i + 1) begin
            @(posedge clk); #1;
            h_prev = dut.hcnt; r_prev = vga_rgb;
            @(posedge clk); #1;
            if (dut.hcnt == h_prev) begin
                if (vga_rgb !== r_prev) viol = viol + 1;
            end else begin
                ok = ok + 1;
            end
        end
        $display("[tb] pixel tick: %0d advances, %0d early rgb changes / 256 clk", ok, viol);
        chk("rgb changes only on the 25 MHz pixel tick (0 violations)", viol == 0);
        chk("raster advances once per 2 clk (256 of 512 clk)", ok == 256);

        $display("[tb] blanking violations: hblank %0d, vblank %0d", hb_err, vb_err);
        chk("hblank rgb == 0 for the whole run (0 violations)", hb_err == 0);
        chk("vblank rgb == 0 for the whole run (0 violations)", vb_err == 0);

        // ================= phase 3: title bars / background ==================
        // one 640 px line of the title area
        get_px(0, 10, cp);
        chk("title bar 0 (x 0..79) = white FFF", cp === C_BAR0_EXP);
        get_px(120, 10, cp);
        chk("title bar 1 (x 80..159) = yellow FF0", cp === 12'hFF0);
        get_px(200, 10, cp);
        chk("title bar 2 (x 160..239) = cyan 0FF", cp === 12'h0FF);
        get_px(280, 10, cp);
        chk("title bar 3 (x 240..319) = green 0F0", cp === 12'h0F0);
        get_px(360, 10, cp);
        chk("title bar 4 (x 320..399) = magenta F0F", cp === 12'hF0F);
        get_px(440, 10, cp);
        chk("title bar 5 (x 400..479) = red F00", cp === 12'hF00);
        get_px(520, 10, cp);
        chk("title bar 6 (x 480..559) = blue 00F", cp === 12'h00F);
        get_px(600, 10, cp);
        chk("title bar 7 (x 560..639) = black 000", cp === 12'h000);

        get_px(79, 10, cp); get_px(80, 10, cq);
        chk("title bar boundary at x=79/80 (white -> yellow)",
            (cp === C_BAR0_EXP) && (cq === 12'hFF0));

        get_px(320, 60, cp); get_px(8, 470, cq);
        chk("background 024 at (320,60) and (8,470)",
            (cp === C_BG) && (cq === C_BG));

        get_px(320, 40, cp); get_px(320, 41, cq);
        chk("separator line at y=40..41 = 444", (cp === C_SEP) && (cq === C_SEP));

        // ================= phase 4: class name, class 0 = ROCK ===============
        // font art: R starts at column 1 -> first ink at x0+4, row 0 y=96..99
        cnt_cls[0] = 0; sig_cls[0] = 64'd0;
        measure_band(TXT_Y, 12'hF00, lx, rx, cnt, sig_cls[0]);
        cnt_cls[0] = cnt;
        get_px(260, 97, cp); get_px(256, 97, cq);
        $display("[tb] class 0 ROCK: ink %0d, bbox %0d..%0d", cnt, lx, rx);
        chk("class 0 ROCK: ink 1088 px, bbox 260..379, colour F00, bg at x0",
            (cnt == 1088) && (lx == 260) && (rx == 379) && (cp === 12'hF00) &&
            (cq === C_BG));

        // ---- IDLE status word (font art: 4 chars, 66 font px -> 1056) ----
        measure_band(STA_Y, 12'h888, lx, rx, cnt, sig_idle);
        get_px(20, 433, cp);
        $display("[tb] status IDLE: ink %0d, bbox %0d..%0d", cnt, lx, rx);
        chk("status IDLE: ink 1056 px, colour 888, bbox 20..139",
            (cnt == 1056) && (lx == 20) && (rx == 139) && (cp === 12'h888));

        // ================= phase 5: class 1 PAPER + score vector A + DONE ====
        // score A: -2432(-38), 0, +6400(+100), max -> bars 218, 256, 356, 512
        cls = 3'd1;
        set_score(-2432, 0, 6400, 262143);
        done = 1'b1; #(CLK_NS); done = 1'b0;       // 1-cycle pulse, must latch
        wait_frame;
        measure_band(TXT_Y, 12'h0F0, lx, rx, cnt, sig_cls[1]);
        cnt_cls[1] = cnt;
        get_px(244, 97, cp); get_px(240, 97, cq);
        $display("[tb] class 1 PAPER: ink %0d, bbox %0d..%0d", cnt, lx, rx);
        chk("class 1 PAPER: ink 1520 px, bbox 244..395, colour 0F0, bg at x0",
            (cnt == 1520) && (lx == 244) && (rx == 395) && (cp === 12'h0F0) &&
            (cq === C_BG));

        measure_bar(0, 12'hF00, len_a[0]);
        measure_bar(1, 12'h0F0, len_a[1]);
        measure_bar(2, 12'h0FF, len_a[2]);
        measure_bar(3, 12'hFA0, len_a[3]);
        $display("[tb] score A bar lengths = %0d %0d %0d %0d (expect %0d %0d %0d %0d)",
                 len_a[0], len_a[1], len_a[2], len_a[3],
                 explen(-2432), explen(0), explen(6400), explen(262143));
        chk("score A bar0 = 218 px (raw -2432)", len_a[0] == explen(-2432));
        chk("score A bar1 = 256 px (raw 0)",     len_a[1] == explen(0));
        chk("score A bar2 = 356 px (raw +6400)", len_a[2] == explen(6400));
        chk("score A bar3 = 512 px (raw max, saturated)", len_a[3] == explen(262143));

        measure_band(STA_Y, 12'h0F0, lx, rx, cnt, sig_done);
        $display("[tb] status DONE: ink %0d, bbox %0d..%0d", cnt, lx, rx);
        chk("status DONE latched from a 1-cycle done pulse (ink 1200, 0F0)",
            (cnt == 1200) && (lx == 20) && (rx == 139));

        // ================= phase 6: class 2 SCISSORS + score vector B =======
        cls = 3'd2;
        set_score(6400, -6400, 0, -262144);
        wait_frame;
        measure_band(TXT_Y, 12'h0FF, lx, rx, cnt, sig_cls[2]);
        cnt_cls[2] = cnt;
        get_px(200, 97, cp); get_px(192, 97, cq);
        $display("[tb] class 2 SCISSORS: ink %0d, bbox %0d..%0d", cnt, lx, rx);
        chk("class 2 SCISSORS: ink 2240 px, bbox 196..443, colour 0FF, bg at x0",
            (cnt == 2240) && (lx == 196) && (rx == 443) && (cp === 12'h0FF) &&
            (cq === C_BG));

        measure_bar(0, 12'hF00, len_b[0]);
        measure_bar(1, 12'h0F0, len_b[1]);
        measure_bar(2, 12'h0FF, len_b[2]);
        measure_bar(3, 12'hFA0, len_b[3]);
        $display("[tb] score B bar lengths = %0d %0d %0d %0d (expect %0d %0d %0d %0d)",
                 len_b[0], len_b[1], len_b[2], len_b[3],
                 explen(6400), explen(-6400), explen(0), explen(-262144));
        chk("score B bar0 = 356 px (raw +6400)", len_b[0] == explen(6400));
        chk("score B bar1 = 156 px (raw -6400)", len_b[1] == explen(-6400));
        chk("score B bar2 = 256 px (raw 0)",     len_b[2] == explen(0));
        chk("score B bar3 = 0 px (raw min, saturated)", len_b[3] == explen(-262144));

        chk("score bars are monotonic in the score (A->B: +, -, -, -)",
            (len_b[0] > len_a[0]) && (len_b[1] < len_a[1]) &&
            (len_b[2] < len_a[2]) && (len_b[3] < len_a[3]));

        dump_ppm("vga_frame_scissors.ppm");        // evidence for the report

        // ================= phase 7: class 3 JUNK + BUSY (err also set) ======
        cls = 3'd3;
        set_score(0, 0, 0, 0);
        busy = 1'b1;
        err  = 1'b1;                               // BUSY must win over ERR
        wait_frame;
        measure_band(TXT_Y, 12'hFA0, lx, rx, cnt, sig_cls[3]);
        cnt_cls[3] = cnt;
        get_px(268, 97, cp); get_px(256, 97, cq);
        $display("[tb] class 3 JUNK: ink %0d, bbox %0d..%0d", cnt, lx, rx);
        chk("class 3 JUNK: ink 1008 px, bbox 260..379, colour FA0, bg at x0",
            (cnt == 1008) && (lx == 260) && (rx == 379) && (cp === 12'hFA0) &&
            (cq === C_BG));

        measure_band(STA_Y, 12'hFF0, lx, rx, cnt, sig_busy);
        $display("[tb] status BUSY: ink %0d, bbox %0d..%0d", cnt, lx, rx);
        chk("status BUSY: ink 1136 px, colour FF0, bbox 20..139",
            (cnt == 1136) && (lx == 20) && (rx == 139));

        // ================= phase 8: ERR (busy released) =====================
        busy = 1'b0;                               // err stays 1
        wait_frame;
        measure_band(STA_Y, 12'hF00, lx, rx, cnt, sig_err);
        get_px(20, 433, cp);
        $display("[tb] status ERR: ink %0d, bbox %0d..%0d", cnt, lx, rx);
        chk("status ERR: ink 976 px, colour F00, bbox 20..107",
            (cnt == 976) && (lx == 20) && (rx == 107) && (cp === 12'hF00));

        // ERR must be gone once the flag drops, and the DONE latched in phase 5
        // must have been cleared by the BUSY of phase 7 -> back to IDLE
        err = 1'b0;
        wait_frame;
        measure_band(STA_Y, 12'h888, lx, rx, cnt, sig_idle);
        $display("[tb] status back to IDLE: ink %0d", cnt);
        chk("BUSY cleared the latched DONE: status returns to IDLE (ink 1056)",
            cnt == 1056);

        // ================= phase 9: four distinct class strings =============
        $display("[tb] class signatures: %h %h %h %h",
                 sig_cls[0], sig_cls[1], sig_cls[2], sig_cls[3]);
        chk("4 class strings give 4 distinct pixel signatures",
            (sig_cls[0] !== sig_cls[1]) && (sig_cls[0] !== sig_cls[2]) &&
            (sig_cls[0] !== sig_cls[3]) && (sig_cls[1] !== sig_cls[2]) &&
            (sig_cls[1] !== sig_cls[3]) && (sig_cls[2] !== sig_cls[3]));

        $display("[tb] status signatures: %h %h %h %h",
                 sig_idle, sig_busy, sig_done, sig_err);
        chk("4 status words give 4 distinct pixel signatures",
            (sig_idle !== sig_busy) && (sig_idle !== sig_done) &&
            (sig_idle !== sig_err) && (sig_busy !== sig_done) &&
            (sig_busy !== sig_err) && (sig_done !== sig_err));

        // ================= phase 10: pixel enable freeze ====================
        // a stopped pixel tick must stop the raster (proves pix_clk_en gates it)
        @(posedge vga_hs);
        pix_en = 1'b0;
        i = hs_edges;
        repeat (2000) @(posedge clk);              // 40 us > one HS period
        $display("[tb] hs edges while pix_clk_en=0: %0d (before %0d)", hs_edges, i);
        chk("pix_clk_en=0 freezes the raster (0 hs edges in 40 us)", hs_edges == i);
        pix_en = 1'b1;
        repeat (2000) @(posedge clk);
        chk("pix_clk_en=1 restarts the raster (hs edges resume)", hs_edges > i);

        // ================= summary ==========================================
        $display("=========================================================");
        if (nfail == 0)
            $display("[RESULT] PASS  %0d checks, all passed", npass);
        else
            $display("[RESULT] FAIL  %0d checks, %0d PASS, %0d FAIL",
                     npass + nfail, npass, nfail);
        $display("=========================================================");
        $finish;
    end

    // watchdog: a hung TB must not run forever
    initial begin
        #(600000000);                              // 600 ms sim time
        $display("[FAIL] watchdog timeout (simulation did not finish)");
        $display("[RESULT] FAIL  watchdog");
        $finish;
    end

endmodule

