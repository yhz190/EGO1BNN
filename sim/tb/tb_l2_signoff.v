// tb_l2_signoff.v -- win_buf(W=8) + L2 (depthwise 3x3) module-level sign-off
// Inputs  : golden L1 activation bitmaps (8 planes x 64x64), 4 frames
// Weights : authoritative L2 weights (8 planes x 9 taps) + integer theta (floor)
// Checked : per-pixel window content (8 planes), per-channel XNOR counts, activations, aggregates
`timescale 1ns / 1ps

module tb_l2_signoff;

    localparam integer NFRAMES = 4;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---------------- DUT ----------------
    reg         sof, feed_en, mon_on;
    wire        din_req, win_valid, done;
    wire [71:0] win;
    wire [5:0]  win_row, win_col;
    reg  [7:0]  din;
    reg  [7:0]  frm [0:4095];          // pixel value = 8 plane bits
    integer     feed_idx;

    win_buf #(.W(8), .CW(65), .CR(65)) u_wb (
        .clk(clk), .rst_n(rst_n), .sof(sof), .din_we(feed_en & din_req), .din(din),
        .win(win), .win_valid(win_valid), .win_row(win_row), .win_col(win_col),
        .din_req(din_req), .done(done)
    );

    always @* din = frm[feed_idx];
    always @(posedge clk or negedge rst_n)
        if (!rst_n) feed_idx <= 0;
        else if (feed_en && din_req) feed_idx <= feed_idx + 1;

    // ---------------- data ----------------
    reg [63:0] rowbits [0:8*64-1];     // [p*64 + r] = row r of plane p
    reg [63:0] tmprow  [0:63];
    reg [71:0] wgt;
    reg [3:0]  theta [0:7];
    reg [63:0] expw  [0:4095];

    integer fd, r, c, i, ch, p, t, k, dummy;
    integer errors, checks;
    integer bad_win, bad_cnt, bad_act, nwin, nreq;
    integer sum_cnt, sum_act, exp_sum_cnt, exp_sum_act;

    function integer bit_at;
        input integer pp, rr, cc;
        begin
            if (rr < 0 || rr > 63 || cc < 0 || cc > 63) bit_at = 0;
            else bit_at = (rowbits[pp*64 + rr] >> (63 - cc)) & 1;
        end
    endfunction

    function [8:0] ref_win;
        input integer pp, rr, cc;
        integer ky, kx;
        begin
            ref_win = 9'd0;
            for (ky = -1; ky <= 1; ky = ky + 1)
                for (kx = -1; kx <= 1; kx = kx + 1)
                    ref_win[(ky+1)*3 + (kx+1)] = bit_at(pp, rr + ky, cc + kx);
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

    reg [8:0] rw;
    reg [3:0] cnt_ch;

    always @(posedge clk) begin
        if (rst_n && mon_on) begin
            if (din_req) nreq = nreq + 1;
            if (win_valid) begin
                nwin = nwin + 1;
                for (ch = 0; ch < 8; ch = ch + 1) begin
                    rw = ref_win(ch, win_row, win_col);
                    if (win[ch*9 +: 9] !== rw) begin
                        if (bad_win < 5)
                            $display("       win mismatch plane%0d (%0d,%0d): got %03x exp %03x",
                                     ch, win_row, win_col, win[ch*9 +: 9], rw);
                        bad_win = bad_win + 1;
                    end
                    cnt_ch = pop9(~(win[ch*9 +: 9] ^ wgt[ch*9 +: 9]));
                    if (cnt_ch !== expw[win_row * 64 + win_col][ch*4 +: 4]) begin
                        if (bad_cnt < 5)
                            $display("       count mismatch ch%0d (%0d,%0d): got %0d exp %0d",
                                     ch, win_row, win_col, cnt_ch,
                                     expw[win_row * 64 + win_col][ch*4 +: 4]);
                        bad_cnt = bad_cnt + 1;
                    end
                    if ((cnt_ch > theta[ch]) !== expw[win_row * 64 + win_col][32 + ch]) begin
                        if (bad_act < 5)
                            $display("       act mismatch ch%0d (%0d,%0d)", ch, win_row, win_col);
                        bad_act = bad_act + 1;
                    end
                    sum_cnt = sum_cnt + cnt_ch;
                    sum_act = sum_act + (cnt_ch > theta[ch]);
                end
            end
        end
    end

    task load_planes;
        input integer k;
        integer pp, rr;
        begin
            for (pp = 0; pp < 8; pp = pp + 1) begin
                case ({k, pp})
                    {0, 0}: $readmemh("data/plane_0_0.hex", tmprow);
                    {0, 1}: $readmemh("data/plane_0_1.hex", tmprow);
                    {0, 2}: $readmemh("data/plane_0_2.hex", tmprow);
                    {0, 3}: $readmemh("data/plane_0_3.hex", tmprow);
                    {0, 4}: $readmemh("data/plane_0_4.hex", tmprow);
                    {0, 5}: $readmemh("data/plane_0_5.hex", tmprow);
                    {0, 6}: $readmemh("data/plane_0_6.hex", tmprow);
                    {0, 7}: $readmemh("data/plane_0_7.hex", tmprow);
                    {1, 0}: $readmemh("data/plane_1_0.hex", tmprow);
                    {1, 1}: $readmemh("data/plane_1_1.hex", tmprow);
                    {1, 2}: $readmemh("data/plane_1_2.hex", tmprow);
                    {1, 3}: $readmemh("data/plane_1_3.hex", tmprow);
                    {1, 4}: $readmemh("data/plane_1_4.hex", tmprow);
                    {1, 5}: $readmemh("data/plane_1_5.hex", tmprow);
                    {1, 6}: $readmemh("data/plane_1_6.hex", tmprow);
                    {1, 7}: $readmemh("data/plane_1_7.hex", tmprow);
                    {2, 0}: $readmemh("data/plane_2_0.hex", tmprow);
                    {2, 1}: $readmemh("data/plane_2_1.hex", tmprow);
                    {2, 2}: $readmemh("data/plane_2_2.hex", tmprow);
                    {2, 3}: $readmemh("data/plane_2_3.hex", tmprow);
                    {2, 4}: $readmemh("data/plane_2_4.hex", tmprow);
                    {2, 5}: $readmemh("data/plane_2_5.hex", tmprow);
                    {2, 6}: $readmemh("data/plane_2_6.hex", tmprow);
                    {2, 7}: $readmemh("data/plane_2_7.hex", tmprow);
                    default: begin
                        case (pp)
                            0: $readmemh("data/plane_3_0.hex", tmprow);
                            1: $readmemh("data/plane_3_1.hex", tmprow);
                            2: $readmemh("data/plane_3_2.hex", tmprow);
                            3: $readmemh("data/plane_3_3.hex", tmprow);
                            4: $readmemh("data/plane_3_4.hex", tmprow);
                            5: $readmemh("data/plane_3_5.hex", tmprow);
                            6: $readmemh("data/plane_3_6.hex", tmprow);
                            default: $readmemh("data/plane_3_7.hex", tmprow);
                        endcase
                    end
                endcase
                for (rr = 0; rr < 64; rr = rr + 1) rowbits[pp*64 + rr] = tmprow[rr];
            end
            for (r = 0; r < 64; r = r + 1)
                for (c = 0; c < 64; c = c + 1) begin
                    frm[r*64 + c] = 8'd0;
                    for (p = 0; p < 8; p = p + 1)
                        frm[r*64 + c][p] = bit_at(p, r, c);
                end
            case (k)
                0: fd = $fopen("data/expect2_0.txt", "r");
                1: fd = $fopen("data/expect2_1.txt", "r");
                2: fd = $fopen("data/expect2_2.txt", "r");
                default: fd = $fopen("data/expect2_3.txt", "r");
            endcase
            if (fd == 0) begin $display("[FAIL] open expect2_%0d.txt", k); errors = errors + 1; end
            exp_sum_cnt = 0; exp_sum_act = 0;
            for (i = 0; i < 4096; i = i + 1) begin
                expw[i] = 64'd0;
                for (ch = 0; ch < 8; ch = ch + 1) begin
                    t = 0; dummy = $fscanf(fd, "%d", t);
                    expw[i][ch*4 +: 4] = t[3:0]; exp_sum_cnt = exp_sum_cnt + t;
                end
                for (ch = 0; ch < 8; ch = ch + 1) begin
                    t = 0; dummy = $fscanf(fd, "%d", t);
                    expw[i][32 + ch] = t[0]; exp_sum_act = exp_sum_act + t;
                end
            end
            $fclose(fd);
        end
    endtask

    task run_frame;
        input integer k;
        begin
            load_planes(k);
            feed_idx = 0; feed_en = 1'b0;
            bad_win = 0; bad_cnt = 0; bad_act = 0;
            nwin = 0; nreq = 0; sum_cnt = 0; sum_act = 0;
            mon_on = 1'b1;
            @(negedge clk); sof = 1'b1; feed_en = 1'b1;
            @(negedge clk); sof = 1'b0;
            wait (done == 1'b1);
            @(negedge clk); @(negedge clk);
            feed_en = 1'b0; mon_on = 1'b0;

            $display("[frame %0d] windows=%0d requests=%0d", k, nwin, nreq);
            if (nwin == 4096 && nreq == 4096 && bad_win == 0) begin
                $display("[PASS] frame %0d : win_buf(W=8) 4096/4096 windows x 8 planes", k); checks = checks + 1;
            end else begin
                $display("[FAIL] frame %0d : windows=%0d req=%0d bad_win=%0d", k, nwin, nreq, bad_win); errors = errors + 1;
            end
            if (bad_cnt == 0) begin
                $display("[PASS] frame %0d : L2 counts all match golden (32768 values)", k); checks = checks + 1;
            end else begin
                $display("[FAIL] frame %0d : %0d count mismatches", k, bad_cnt); errors = errors + 1;
            end
            if (bad_act == 0) begin
                $display("[PASS] frame %0d : L2 activations all match golden", k); checks = checks + 1;
            end else begin
                $display("[FAIL] frame %0d : %0d activation mismatches", k, bad_act); errors = errors + 1;
            end
            if (sum_cnt === exp_sum_cnt && sum_act === exp_sum_act) begin
                $display("[PASS] frame %0d : aggregates count=%0d act=%0d (python)", k, sum_cnt, sum_act); checks = checks + 1;
            end else begin
                $display("[FAIL] frame %0d : aggregates count=%0d/%0d act=%0d/%0d", k, sum_cnt, exp_sum_cnt, sum_act, exp_sum_act); errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0; checks = 0;
        fd = $fopen("data/w2.hex", "r");
        if (fd == 0) begin $display("[FAIL] open w2.hex"); $finish; end
        dummy = $fscanf(fd, "%h", wgt);
        $fclose(fd);
        fd = $fopen("data/theta2.hex", "r");
        for (ch = 0; ch < 8; ch = ch + 1) begin
            t = 0; dummy = $fscanf(fd, "%h", t); theta[ch] = t[3:0];
        end
        $fclose(fd);
        $display("[cfg] w2 = %018h theta = %0d %0d %0d %0d %0d %0d %0d %0d",
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
