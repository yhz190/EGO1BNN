// tb_l3_signoff.v -- L3 (1x1 conv, 8 in -> 4 out) module-level sign-off
// Holds the real RTL blocks used by L3: popcnt9 (as an 8-bit popcount via {1'b0,eq})
// and theta_cmp (4 of its 8 channels). No window buffer: L3 is 1x1 (patch1: no extra cycles).
//
// Inputs  : golden L2 activation bitmaps (8 planes x 64x64), 4 frames
// Weights : authoritative L3 weights (4 out x 8 in) + integer theta (floor)
// Checked : per-pixel counts (4 ch) and activations vs golden.npz c3 / L3
`timescale 1ns / 1ps

module tb_l3_signoff;

    localparam integer NFRAMES = 4;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---------------- RTL under test ----------------
    reg  [8:0]  pv;
    wire [3:0]  pcnt;

    popcnt9 u_pc (.din(pv), .cnt(pcnt));

    reg         th_we;
    reg  [2:0]  th_ch;
    reg  [3:0]  th_data;
    wire [7:0]  act;
    wire [31:0] th_val;

    reg  [7*12:0] acc_lo;          // only channels 0..3 are meaningful
    wire [95:0]  acc_all = {64'd0, acc_lo};

    theta_cmp #(.W_TH(4), .W_ACC(12)) u_th (
        .clk(clk), .rst_n(rst_n), .th_we(th_we), .th_ch(th_ch), .th_data(th_data),
        .acc(acc_all), .act(act), .th_val(th_val)
    );

    // ---------------- data ----------------
    reg [63:0] rowbits [0:8*64-1];
    reg [63:0] tmprow  [0:63];
    reg [31:0] w3_q;                // 4 out x 8 in
    reg [3:0]  theta [0:3];
    reg [31:0] expw  [0:4095];      // [15:0] counts (4 x 4b), [23:16] acts (4 x 1b)

    integer fd, r, c, ch, p, t, k, i, dummy;
    integer errors, checks;
    integer bad_cnt, bad_act;
    integer sum_cnt, sum_act, exp_sum_cnt, exp_sum_act;

    function integer bit_at;
        input integer pp, rr, cc;
        begin
            if (rr < 0 || rr > 63 || cc < 0 || cc > 63) bit_at = 0;
            else bit_at = (rowbits[pp*64 + rr] >> (63 - cc)) & 1;
        end
    endfunction

    task load_planes;
        input integer k;
        integer pp, rr;
        begin
            for (pp = 0; pp < 8; pp = pp + 1) begin
                case ({k, pp})
                    {0,0}: $readmemh("data/in_0_0.hex", tmprow);
                    {0,1}: $readmemh("data/in_0_1.hex", tmprow);
                    {0,2}: $readmemh("data/in_0_2.hex", tmprow);
                    {0,3}: $readmemh("data/in_0_3.hex", tmprow);
                    {0,4}: $readmemh("data/in_0_4.hex", tmprow);
                    {0,5}: $readmemh("data/in_0_5.hex", tmprow);
                    {0,6}: $readmemh("data/in_0_6.hex", tmprow);
                    {0,7}: $readmemh("data/in_0_7.hex", tmprow);
                    {1,0}: $readmemh("data/in_1_0.hex", tmprow);
                    {1,1}: $readmemh("data/in_1_1.hex", tmprow);
                    {1,2}: $readmemh("data/in_1_2.hex", tmprow);
                    {1,3}: $readmemh("data/in_1_3.hex", tmprow);
                    {1,4}: $readmemh("data/in_1_4.hex", tmprow);
                    {1,5}: $readmemh("data/in_1_5.hex", tmprow);
                    {1,6}: $readmemh("data/in_1_6.hex", tmprow);
                    {1,7}: $readmemh("data/in_1_7.hex", tmprow);
                    {2,0}: $readmemh("data/in_2_0.hex", tmprow);
                    {2,1}: $readmemh("data/in_2_1.hex", tmprow);
                    {2,2}: $readmemh("data/in_2_2.hex", tmprow);
                    {2,3}: $readmemh("data/in_2_3.hex", tmprow);
                    {2,4}: $readmemh("data/in_2_4.hex", tmprow);
                    {2,5}: $readmemh("data/in_2_5.hex", tmprow);
                    {2,6}: $readmemh("data/in_2_6.hex", tmprow);
                    {2,7}: $readmemh("data/in_2_7.hex", tmprow);
                    default: begin
                        case (pp)
                            0: $readmemh("data/in_3_0.hex", tmprow);
                            1: $readmemh("data/in_3_1.hex", tmprow);
                            2: $readmemh("data/in_3_2.hex", tmprow);
                            3: $readmemh("data/in_3_3.hex", tmprow);
                            4: $readmemh("data/in_3_4.hex", tmprow);
                            5: $readmemh("data/in_3_5.hex", tmprow);
                            6: $readmemh("data/in_3_6.hex", tmprow);
                            default: $readmemh("data/in_3_7.hex", tmprow);
                        endcase
                    end
                endcase
                for (rr = 0; rr < 64; rr = rr + 1) rowbits[pp*64 + rr] = tmprow[rr];
            end
            case (k)
                0: fd = $fopen("data/expect3_0.txt", "r");
                1: fd = $fopen("data/expect3_1.txt", "r");
                2: fd = $fopen("data/expect3_2.txt", "r");
                default: fd = $fopen("data/expect3_3.txt", "r");
            endcase
            if (fd == 0) begin $display("[FAIL] open expect3_%0d.txt", k); errors = errors + 1; end
            exp_sum_cnt = 0; exp_sum_act = 0;
            for (i = 0; i < 4096; i = i + 1) begin
                expw[i] = 32'd0;
                for (ch = 0; ch < 4; ch = ch + 1) begin
                    t = 0; dummy = $fscanf(fd, "%d", t);
                    expw[i][ch*4 +: 4] = t[3:0]; exp_sum_cnt = exp_sum_cnt + t;
                end
                for (ch = 0; ch < 4; ch = ch + 1) begin
                    t = 0; dummy = $fscanf(fd, "%d", t);
                    expw[i][16 + ch] = t[0]; exp_sum_act = exp_sum_act + t;
                end
            end
            $fclose(fd);
        end
    endtask

    task run_frame;
        input integer k;
        integer x8, e, cc, got;
        begin
            load_planes(k);
            bad_cnt = 0; bad_act = 0; sum_cnt = 0; sum_act = 0;
            for (r = 0; r < 64; r = r + 1) begin
                for (c = 0; c < 64; c = c + 1) begin
                    x8 = 0;
                    for (p = 0; p < 8; p = p + 1) x8 = x8 | (bit_at(p, r, c) << p);
                    acc_lo = 28'd0;
                    for (cc = 0; cc < 4; cc = cc + 1) begin
                        e = ~((x8 ^ ((w3_q >> (cc*8)) & 8'hFF)) & 8'hFF) & 8'hFF;
                        pv = {1'b0, e[7:0]};
                        #1;
                        got = pcnt;
                        acc_lo[cc*12 +: 12] = got[11:0];
                        sum_cnt = sum_cnt + got;
                        if (got[3:0] !== expw[r*64 + c][cc*4 +: 4]) begin
                            if (bad_cnt < 5)
                                $display("       count mismatch ch%0d (%0d,%0d): got %0d exp %0d",
                                         cc, r, c, got, expw[r*64 + c][cc*4 +: 4]);
                            bad_cnt = bad_cnt + 1;
                        end
                    end
                    #1;
                    for (cc = 0; cc < 4; cc = cc + 1) begin
                        if (act[cc] !== expw[r*64 + c][16 + cc]) begin
                            if (bad_act < 5)
                                $display("       act mismatch ch%0d (%0d,%0d): got %0d exp %0d",
                                         cc, r, c, act[cc], expw[r*64 + c][16 + cc]);
                            bad_act = bad_act + 1;
                        end
                        sum_act = sum_act + act[cc];
                    end
                end
            end
            $display("[frame %0d] pixels=4096", k);
            if (bad_cnt == 0) begin
                $display("[PASS] frame %0d : L3 counts all match golden (16384 values)", k); checks = checks + 1;
            end else begin
                $display("[FAIL] frame %0d : %0d count mismatches", k, bad_cnt); errors = errors + 1;
            end
            if (bad_act == 0) begin
                $display("[PASS] frame %0d : L3 activations all match golden", k); checks = checks + 1;
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
        fd = $fopen("data/w3.hex", "r");
        if (fd == 0) begin $display("[FAIL] open w3.hex"); $finish; end
        dummy = $fscanf(fd, "%h", w3_q);
        $fclose(fd);
        fd = $fopen("data/theta3.hex", "r");
        for (ch = 0; ch < 4; ch = ch + 1) begin
            t = 0; dummy = $fscanf(fd, "%h", t); theta[ch] = t[3:0];
        end
        $fclose(fd);
        $display("[cfg] w3 = %08h theta = %0d %0d %0d %0d", w3_q, theta[0], theta[1], theta[2], theta[3]);

        rst_n = 1'b0; th_we = 1'b0; th_ch = 3'd0; th_data = 4'd0; pv = 9'd0;
        #50 rst_n = 1'b1;
        // load theta into theta_cmp (channels 0..3)
        for (ch = 0; ch < 4; ch = ch + 1) begin
            @(negedge clk); th_we = 1'b1; th_ch = ch[2:0]; th_data = theta[ch];
        end
        @(negedge clk); th_we = 1'b0;
        #20;
        for (k = 0; k < NFRAMES; k = k + 1) run_frame(k);
        $display("==================================================");
        $display("[RESULT] %0s (%0d checks, %0d errors)", (errors == 0) ? "ALL PASS" : "FAIL", checks, errors);
        $finish;
    end

endmodule
