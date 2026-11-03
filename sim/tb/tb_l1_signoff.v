// tb_l1_signoff.v -- win_buf + L1 module-level sign-off against the T8 golden vectors
//
// Checks, for 4 real frames (rock_000..003) with the authoritative checkpoint's L1
// weights and integer theta:
//   1) win_buf window content per pixel (against a TB-side reference built from the frame)
//   2) per-channel XNOR match counts (against golden.npz c1)
//   3) per-channel activation = (count > theta[c]) (against golden.npz L1 bitmaps)
//   4) total count sum / activation sum (against Python aggregates)
//   5) din_req count = 4096 and window count = 4096
//
// Window bit order (win_buf / lane_array): win[ky*3+kx], ky=0 row-above, kx=0 col-left.
// Padding: bit-domain 0 (== -1 in the +-1 domain), i.e. out-of-frame taps are 0.
`timescale 1ns / 1ps

module tb_l1_signoff;

    localparam integer NFRAMES = 4;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---------------- DUT: win_buf, single plane (L1 path) ----------------
    reg         sof, feed_en;
    reg         mon_on;
    wire        din_req, win_valid, done;
    wire [8:0]  win;
    wire [5:0]  win_row, win_col;
    reg  [0:0]  din;
    reg  [0:0]  frm [0:4095];
    integer     feed_idx;

    win_buf #(.W(1), .CW(65), .CR(65)) u_wb (
        .clk(clk), .rst_n(rst_n), .sof(sof), .din_we(feed_en & din_req), .din(din),
        .win(win), .win_valid(win_valid), .win_row(win_row), .win_col(win_col),
        .din_req(din_req), .done(done)
    );

    always @* din = frm[feed_idx];
    always @(posedge clk or negedge rst_n)
        if (!rst_n) feed_idx <= 0;
        else if (feed_en && din_req) feed_idx <= feed_idx + 1;

    // ---------------- stimulus data ----------------
    reg [63:0] rowbits [0:63];          // one row of the frame, MSB = column 0
    reg [71:0] wgt;                     // L1 weights, lane c -> wgt[c*9 +: 9]
    reg [3:0]  theta [0:7];
    reg [63:0] expw  [0:4095];          // [31:0] counts (8 x 4b), [39:32] acts (8 x 1b)

    integer fd, r, c, i, ch, t, k, dummy;
    reg [71:0] wtmp;
    integer errors, checks;
    integer bad_win, bad_cnt, bad_act, nwin, nreq;
    integer sum_cnt, sum_act, exp_sum_cnt, exp_sum_act;

    // ---------------- TB-side reference ----------------
    function integer bit_at;            // frame bit at (rr, cc), 0 outside
        input integer rr, cc;
        begin
            if (rr < 0 || rr > 63 || cc < 0 || cc > 63) bit_at = 0;
            else bit_at = (rowbits[rr] >> (63 - cc)) & 1;
        end
    endfunction

    function [8:0] ref_win;
        input integer rr, cc;
        integer ky, kx;
        begin
            ref_win = 9'd0;
            for (ky = -1; ky <= 1; ky = ky + 1)
                for (kx = -1; kx <= 1; kx = kx + 1)
                    ref_win[(ky+1)*3 + (kx+1)] = bit_at(rr + ky, cc + kx);
        end
    endfunction

    function [3:0] pop9;
        input [8:0] v;
        integer j;
        begin
            pop9 = 4'd0;
            for (j = 0; j < 9; j = j + 1) pop9 = pop9 + v[j];
        end
    endfunction

    // ---------------- monitors ----------------
    reg [8:0]  rw;
    reg [3:0]  cnt_ch;
    reg        act_ch;

    always @(posedge clk) begin
        if (rst_n && mon_on) begin
            if (din_req) nreq = nreq + 1;
            if (win_valid) begin
                nwin = nwin + 1;
                rw = ref_win(win_row, win_col);
                if (win !== rw[8:0]) begin
                    if (bad_win < 5)
                        $display("       win mismatch (%0d,%0d): got %03x exp %03x",
                                 win_row, win_col, win, rw);
                    bad_win = bad_win + 1;
                end
                for (ch = 0; ch < 8; ch = ch + 1) begin
                    cnt_ch = pop9(~(win ^ wgt[ch*9 +: 9]));
                    act_ch = (cnt_ch > theta[ch]);
                    if (cnt_ch !== expw[win_row * 64 + win_col][ch*4 +: 4]) begin
                        if (bad_cnt < 5)
                            $display("       count mismatch ch%0d (%0d,%0d): got %0d exp %0d",
                                     ch, win_row, win_col, cnt_ch,
                                     expw[win_row * 64 + win_col][ch*4 +: 4]);
                        bad_cnt = bad_cnt + 1;
                    end
                    if (act_ch !== expw[win_row * 64 + win_col][32 + ch]) begin
                        if (bad_act < 5)
                            $display("       act mismatch ch%0d (%0d,%0d): got %0d exp %0d",
                                     ch, win_row, win_col, act_ch,
                                     expw[win_row * 64 + win_col][32 + ch]);
                        bad_act = bad_act + 1;
                    end
                    sum_cnt = sum_cnt + cnt_ch;
                    sum_act = sum_act + act_ch;
                end
            end
        end
    end

    // ---------------- load + run ----------------
    task load_frame;
        input integer k;
        begin
            case (k)
                0: $readmemh("data/frame_0.hex", rowbits);
                1: $readmemh("data/frame_1.hex", rowbits);
                2: $readmemh("data/frame_2.hex", rowbits);
                default: $readmemh("data/frame_3.hex", rowbits);
            endcase
            for (r = 0; r < 64; r = r + 1)
                for (c = 0; c < 64; c = c + 1)
                    frm[r * 64 + c] = (rowbits[r] >> (63 - c)) & 1;
            case (k)
                0: fd = $fopen("data/expect_0.txt", "r");
                1: fd = $fopen("data/expect_1.txt", "r");
                2: fd = $fopen("data/expect_2.txt", "r");
                default: fd = $fopen("data/expect_3.txt", "r");
            endcase
            if (fd == 0) begin
                $display("[FAIL] cannot open expect_%0d.txt", k);
                errors = errors + 1;
            end
            exp_sum_cnt = 0; exp_sum_act = 0;
            for (i = 0; i < 4096; i = i + 1) begin
                expw[i] = 64'd0;
                for (ch = 0; ch < 8; ch = ch + 1) begin
                    t = 0;
                    dummy = $fscanf(fd, "%d", t);
                    expw[i][ch*4 +: 4] = t[3:0];
                    exp_sum_cnt = exp_sum_cnt + t;
                end
                for (ch = 0; ch < 8; ch = ch + 1) begin
                    t = 0;
                    dummy = $fscanf(fd, "%d", t);
                    expw[i][32 + ch] = t[0];
                    exp_sum_act = exp_sum_act + t;
                end
            end
            $fclose(fd);
        end
    endtask

    task run_frame;
        input integer k;
        begin
            load_frame(k);
            feed_idx = 0; feed_en = 1'b0;
            bad_win = 0; bad_cnt = 0; bad_act = 0;
            nwin = 0; nreq = 0; sum_cnt = 0; sum_act = 0;
            mon_on = 1'b1;
            @(negedge clk); sof = 1'b1; feed_en = 1'b1;
            @(negedge clk); sof = 1'b0;
            wait (done == 1'b1);
            @(negedge clk);
            @(negedge clk);
            feed_en = 1'b0;
            mon_on = 1'b0;

            $display("[frame %0d] windows=%0d requests=%0d", k, nwin, nreq);
            if (nwin == 4096 && nreq == 4096 && bad_win == 0) begin
                $display("[PASS] frame %0d : win_buf 4096/4096 windows, 4096 requests", k);
                checks = checks + 1;
            end else begin
                $display("[FAIL] frame %0d : windows=%0d requests=%0d bad_win=%0d",
                         k, nwin, nreq, bad_win);
                errors = errors + 1;
            end
            if (bad_cnt == 0) begin
                $display("[PASS] frame %0d : L1 counts all match golden (%0d values)", k, 4096 * 8);
                checks = checks + 1;
            end else begin
                $display("[FAIL] frame %0d : %0d count mismatches", k, bad_cnt);
                errors = errors + 1;
            end
            if (bad_act == 0) begin
                $display("[PASS] frame %0d : L1 activations all match golden", k);
                checks = checks + 1;
            end else begin
                $display("[FAIL] frame %0d : %0d activation mismatches", k, bad_act);
                errors = errors + 1;
            end
            if (sum_cnt === exp_sum_cnt && sum_act === exp_sum_act) begin
                $display("[PASS] frame %0d : aggregates count=%0d act=%0d (python)", k, sum_cnt, sum_act);
                checks = checks + 1;
            end else begin
                $display("[FAIL] frame %0d : aggregates count=%0d/%0d act=%0d/%0d",
                         k, sum_cnt, exp_sum_cnt, sum_act, exp_sum_act);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0; checks = 0;
        // weights + theta
        fd = $fopen("data/w1.hex", "r");
        if (fd == 0) begin $display("[FAIL] open w1.hex"); $finish; end
        t = 0;
        dummy = $fscanf(fd, "%h", wtmp);
        wgt = wtmp;
        $fclose(fd);
        fd = $fopen("data/theta.hex", "r");
        for (ch = 0; ch < 8; ch = ch + 1) begin
            t = 0;
            dummy = $fscanf(fd, "%h", t);
            theta[ch] = t[3:0];
        end
        $fclose(fd);
        $display("[cfg] wgt = %018h  theta = %0d %0d %0d %0d %0d %0d %0d %0d",
                 wgt, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7]);

        rst_n = 1'b0;
        #50 rst_n = 1'b1;
        #50;

        for (k = 0; k < NFRAMES; k = k + 1) run_frame(k);

        $display("==================================================");
        $display("[RESULT] %0s (%0d checks, %0d errors)", (errors == 0) ? "ALL PASS" : "FAIL", checks, errors);
        $finish;
    end

endmodule
