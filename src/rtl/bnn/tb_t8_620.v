// =====================================================================================
// tb_t8_620.v -- T8 acceptance at RTL level: 620 real frames through bnn_engine,
//                every Q6 class score compared bit-exactly with the golden model.
//
// The TB is strictly black-box / read-only w.r.t. the DUT: it only drives the frozen
// bnn_engine ports (reset, weight blob, FRAME write port, start) and samples
// score_o / cls_o / all_done.  No hierarchical force, no RTL modification.
//
// Data files (read by $readmemh from the simulation working directory):
//   frame_words.hex    620 x 128 x 32 bit, one word per line.  Word i of a frame holds
//                      bytes 4i..4i+3 little endian, i.e. the 512 B packbits frame of
//                      data/vectors_t1a.  Byte b bit 7..0 -> pixels 8b .. 8b+7
//                      (MSB first, identical to preprocess.py np.packbits and to the
//                      top_bnn.v capture bit serialiser).
//   tb_t8_expected.hex 620 lines, one 80 bit word per frame:
//                        [79:76] pred (0..3, 4 pad bits)   [75:57] f3
//                        [56:38] f2   [37:19] f1   [18:0] f0     19-bit two's complement
//   weights.hex        271 word v5 beat-major blob (CRC16 0xB8C7), streamed in 271
//                      consecutive cycles on w_we/w_addr/w_data.
//
// Per frame: stream the 4096 FRAME bits into bank 0 (1 bit/cycle), pulse start, wait
// for the 1-cycle all_done, then compare score_o[75:0] === expected[75:0].
// Frame latency must be 16,911 state cycles (L1..L5 + done).
//
// Usage (from a directory containing the three hex files and the compiled snapshot):
//   xvlog <all rtl/bnn/*.v> tb_t8_620.v
//   xelab -L unisims_ver tb_t8_620 -s tb_t8_620
//   xsim tb_t8_620 -R                              # 620 frames (default)
//   echo 20 > tb_t8_nframes.txt && xsim tb_t8_620 -R   # quick 20-frame check
//
// The frame count is the parameter NFRAMES (default 620); the optional file
// tb_t8_nframes.txt (single integer) overrides it at run time for a quick check.
// (The Vivado .bat loader splits command-line arguments on '=', so a +NFRAMES=v
//  plusarg cannot be passed through it; the file is the official override path.)
//
// Prints [PASS]/[FAIL] and "[RESULT] X/620".
// =====================================================================================
`timescale 1ns / 1ps

module tb_t8_620;

    parameter integer NFRAMES = 620;

    localparam integer MAX_FRAMES = 620;
    localparam integer N_WORDS    = 128;    // 128 x 32 bit = 512 B per frame
    localparam integer EXP_FRAME  = 16911;  // L1..L5 + done state cycles

    // ------------------------------------------------------------------ clock / reset
    reg clk;
    initial clk = 1'b0;
    always #5 clk = ~clk;                   // 100 MHz

    reg rst_n;

    // ------------------------------------------------------------------ DUT ports
    reg         start;
    wire        busy, all_done, ready;
    reg         frame_we;
    reg  [11:0] frame_addr;
    reg         frame_bit;
    reg         frame_bank;
    reg         frame_rd_bank;
    reg         w_we;
    reg  [8:0]  w_addr;
    reg  [31:0] w_data;
    reg         th_we;
    reg  [2:0]  th_sel;
    reg  [2:0]  th_ch;
    reg  [15:0] th_data;
    wire [2:0]  cls_o;
    wire [75:0] score_o;
    wire [47:0] acc_o;
    wire [2:0]  layer_o, state_o;
    wire [12:0] beat_o;
    wire        wload_busy;

    bnn_engine dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start),
        .busy          (busy),
        .all_done      (all_done),
        .ready         (ready),
        .frame_we      (frame_we),
        .frame_addr    (frame_addr),
        .frame_bit     (frame_bit),
        .frame_bank    (frame_bank),
        .frame_rd_bank (frame_rd_bank),
        .w_we          (w_we),
        .w_addr        (w_addr),
        .w_data        (w_data),
        .th_we         (th_we),
        .th_sel        (th_sel),
        .th_ch         (th_ch),
        .th_data       (th_data),
        .cls_o         (cls_o),
        .score_o       (score_o),
        .acc_o         (acc_o),
        .layer_o       (layer_o),
        .state_o       (state_o),
        .beat_o        (beat_o),
        .wload_busy    (wload_busy)
    );

    // ------------------------------------------------------------------ data memories
    reg [31:0] fw_mem [0:MAX_FRAMES*N_WORDS-1];   // 79,360 words
    reg [79:0] ex_mem [0:MAX_FRAMES-1];           // 620 expected score words
    reg [31:0] wblob  [0:270];                    // 271 word v5 weight blob

    // ------------------------------------------------------------------ bookkeeping
    integer nframes, tmp, fr;
    integer k, pp, wi, bi;
    integer npass, nfail, ncls_ok, ncls_bad, first_bad;
    integer cbad0, cbad1, cbad2, cbad3;
    integer lat, lat_min, lat_max;
    integer fd;

    // 19-bit two's complement -> integer for %d display only (comparison is bit-exact)
    function integer s19i;
        input [18:0] v;
        begin
            s19i = v[18] ? (v - 524288) : v;
        end
    endfunction

    // ------------------------------------------------------------------ watchdog
    initial begin
        #400000000;                 // 400 ms at timescale 1ns = 40 M cycles
        $display("[FAIL] watchdog: simulation did not finish in 400 ms sim time (%0t)", $time);
        $finish;
    end

    // ------------------------------------------------------------------ tasks
    // stream the 4096 FRAME bits of frame `fi` into bank 0, 1 bit per cycle
    task write_frame;
        input integer fi;
        integer p;
        integer w;
        integer b;
        begin
            for (p = 0; p < 4096; p = p + 1) begin
                w = fi * N_WORDS + (p >> 5);                 // 32-bit word of this frame
                b = 8 * ((p >> 3) & 3) + (7 - (p & 7));      // MSB first inside each byte
                frame_we   = 1'b1;
                frame_addr = p[11:0];
                frame_bit  = (fw_mem[w] >> b) & 1'b1;
                @(negedge clk);
            end
            frame_we  = 1'b0;
            frame_bit = 1'b0;
            frame_addr = 12'd0;
        end
    endtask

    // ------------------------------------------------------------------ main sequence
    initial begin
        // defaults / optional quick-check frame count
        nframes = NFRAMES;
        tmp     = NFRAMES;
        fr      = 0;
        fd = $fopen("tb_t8_nframes.txt", "r");
        if (fd != 0) begin
            fr = $fscanf(fd, "%d", tmp);
            $fclose(fd);
            $display("[INFO] tb_t8_nframes.txt: fscanf ret=%0d value=%0d", fr, tmp);
        end
        if (fr == 1) nframes = tmp;
        if (nframes < 1)          nframes = 1;
        if (nframes > MAX_FRAMES) nframes = MAX_FRAMES;

        npass = 0; nfail = 0; ncls_ok = 0; ncls_bad = 0; first_bad = -1;
        cbad0 = 0; cbad1 = 0; cbad2 = 0; cbad3 = 0;
        lat_min = 0; lat_max = 0;

        $display("================================================================");
        $display("[INFO] tb_t8_620: %0d frame(s)  (NFRAMES=%0d)", nframes, NFRAMES);
        $display("[INFO] data: frame_words.hex / tb_t8_expected.hex / weights.hex");

        fd = $fopen("frame_words.hex", "r");
        if (fd == 0) begin
            $display("[FAIL] cannot open frame_words.hex (run xsim in the data directory)");
            $finish;
        end
        $fclose(fd);
        fd = $fopen("tb_t8_expected.hex", "r");
        if (fd == 0) begin
            $display("[FAIL] cannot open tb_t8_expected.hex");
            $finish;
        end
        $fclose(fd);
        fd = $fopen("weights.hex", "r");
        if (fd == 0) begin
            $display("[FAIL] cannot open weights.hex");
            $finish;
        end
        $fclose(fd);

        $readmemh("frame_words.hex",    fw_mem);
        $readmemh("tb_t8_expected.hex", ex_mem);
        $readmemh("weights.hex",        wblob);

        // sanity: golden data must be defined
        if (^fw_mem[0] === 1'bx || ^ex_mem[0] === 1'bx || ^wblob[0] === 1'bx) begin
            $display("[FAIL] data files not loaded correctly (X in memory)");
            $finish;
        end

        // ---------------- reset ----------------
        rst_n          = 1'b0;
        start          = 1'b0;
        frame_we       = 1'b0;
        frame_addr     = 12'd0;
        frame_bit      = 1'b0;
        frame_bank     = 1'b0;
        frame_rd_bank  = 1'b0;
        w_we           = 1'b0;
        w_addr         = 9'd0;
        w_data         = 32'd0;
        th_we          = 1'b0;
        th_sel         = 3'd0;
        th_ch          = 3'd0;
        th_data        = 16'd0;
        repeat (8) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // ---------------- 271 word weight / theta blob: 271 consecutive cycles ----------------
        for (k = 0; k < 271; k = k + 1) begin
            w_we   = 1'b1;
            w_addr = k[8:0];
            w_data = wblob[k];
            @(negedge clk);
        end
        w_we   = 1'b0;
        w_addr = 9'd0;
        w_data = 32'd0;

        while (!ready) @(negedge clk);
        $display("[INFO] weight blob loaded (271 words + theta push), ready=1 at %0t", $time);
        repeat (2) @(negedge clk);

        // ---------------- frame loop ----------------
        for (k = 0; k < nframes; k = k + 1) begin
            write_frame(k);                      // 4096 cycles, bank 0

            frame_rd_bank = 1'b0;                // read the bank we just wrote
            start = 1'b1;                        // 1-cycle pulse
            @(negedge clk);
            start = 1'b0;

            lat = 1;
            while (!all_done) begin
                @(negedge clk);
                lat = lat + 1;
            end
            // all_done is high for exactly this cycle and score_o is already stable
            if (k == 0 || lat < lat_min) lat_min = lat;
            if (k == 0 || lat > lat_max) lat_max = lat;

            if (score_o === ex_mem[k][75:0]) begin
                npass = npass + 1;
            end else begin
                nfail = nfail + 1;
                if (first_bad < 0) first_bad = k;
                if (score_o[18:0] !== ex_mem[k][18:0]) cbad0 = cbad0 + 1;
                if (score_o[37:19] !== ex_mem[k][37:19]) cbad1 = cbad1 + 1;
                if (score_o[56:38] !== ex_mem[k][56:38]) cbad2 = cbad2 + 1;
                if (score_o[75:57] !== ex_mem[k][75:57]) cbad3 = cbad3 + 1;
                if (nfail <= 10) begin
                    $display("[FAIL] frame %0d (latency %0d): score_o=%019h expected=%019h xor=%019h",
                             k, lat, score_o, ex_mem[k][75:0],
                             score_o ^ ex_mem[k][75:0]);
                    $display("       exp f0..f3 = %0d %0d %0d %0d   pred=%0d",
                             s19i(ex_mem[k][18:0]),  s19i(ex_mem[k][37:19]),
                             s19i(ex_mem[k][56:38]), s19i(ex_mem[k][75:57]),
                             ex_mem[k][78:76]);
                    $display("       got f0..f3 = %0d %0d %0d %0d   cls_o=%0d",
                             s19i(score_o[18:0]),  s19i(score_o[37:19]),
                             s19i(score_o[56:38]), s19i(score_o[75:57]), cls_o);
                end
            end

            if (cls_o !== ex_mem[k][78:76]) ncls_bad = ncls_bad + 1;
            else                            ncls_ok  = ncls_ok  + 1;

            if (((k + 1) % 50) == 0 || k == nframes - 1) begin
                $display("[INFO] %0d/%0d frames done: score-exact %0d, mismatch %0d, t=%0t",
                         k + 1, nframes, npass, nfail, $time);
            end
        end

        // ---------------- summary ----------------
        $display("================================================================");
        $display("[INFO] frames run          : %0d", nframes);
        $display("[INFO] score-exact frames  : %0d", npass);
        $display("[INFO] mismatched frames   : %0d", nfail);
        if (first_bad >= 0) $display("[INFO] first mismatch frame: %0d", first_bad);
        $display("[INFO] per-class mismatches : f0=%0d f1=%0d f2=%0d f3=%0d",
                 cbad0, cbad1, cbad2, cbad3);
        $display("[INFO] cls_o == pred        : %0d/%0d", ncls_ok, nframes);
        $display("[INFO] frame latency        : %0d..%0d cycles (expect %0d)",
                 lat_min, lat_max, EXP_FRAME);
        if (lat_min != EXP_FRAME || lat_max != EXP_FRAME)
            $display("[WARN] frame latency differs from the 16,911 cycle budget");
        $display("[RESULT] %0d/%0d", npass, nframes);
        if (nfail == 0 && ncls_bad == 0)
            $display("[PASS] T8 RTL regression: %0d/%0d frames bit-exact", npass, nframes);
        else if (nfail == 0)
            $display("[FAIL] T8 RTL regression: scores exact but %0d cls_o mismatch(es)", ncls_bad);
        else
            $display("[FAIL] T8 RTL regression: %0d/%0d frames differ (first %0d)",
                     nfail, nframes, first_bad);
        $display("================================================================");
        $finish;
    end

endmodule
