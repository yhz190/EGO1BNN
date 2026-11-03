// tb_engine.v -- self-checking testbench for the BNN engine skeleton
//
//   test 1 : popcnt9, all 512 input combinations
//   test 2 : lane_array vs Python golden vectors + random cross-check
//   test 3 : win_buf W=1, 4 frames x 4096 windows vs array reference model,
//            explicit padding windows, din_req / window count / window-sum checks
//   test 4 : win_buf W=8 (L2 path), 4096 windows x 8 planes vs array reference model
//   test 5 : accum, directed (reset / en / clr / clr+en / ld)
//   test 6 : theta_cmp, directed (theta load + strict-greater compare)
//   test 7 : integrated L1 chain (win_buf -> lane_array -> accum -> theta_cmp),
//            4096 pixels x 8 channels vs the reference model AND vs Python aggregates
//
// Golden numbers (window sums, lane counts, plane aggregates) were computed by
// independent Python 3.11 scripts (pure integer, no numpy) -- see the record
// sim/20261101_引擎骨架记录.md for the exact formulas.
//
// Run: xvlog popcnt9.v lane_array.v accum.v theta_cmp.v win_buf.v tb_engine.v
//      xelab tb_engine -s sim && xsim sim -runall
`timescale 1ns / 1ps

module tb_engine;

    // ------------------------------------------------------------------ Python goldens
    // lane_array vector A: win/wgt from random seed 20261101, cnt = popcount(XNOR)
    localparam [71:0] WIN_A = 72'h242a0e019b1b247779;
    localparam [71:0] WGT_A = 72'h72101194d0b07bd82e;
    localparam [31:0] CNT_A = 32'h55353323;              // lanes 3,2,3,3,5,3,5,5
    // L1 weights for the integrated test (8 ch x 9 tap, random seed 777)
    localparam [71:0] WGT_L1 = 72'hd4a2a4cbcfdb91c875;

    // ------------------------------------------------------------------ clock / reset
    reg clk;
    reg rst_n;
    initial clk = 1'b0;
    always #5 clk = ~clk;                     // 100 MHz

    integer errors;
    integer checks;

    // ================================================================== helpers
    task chk(input [8*56-1:0] name, input integer got, input integer exp);
        begin
            if (got !== exp) begin
                $display("[FAIL] %0s : got %0d, exp %0d", name, got, exp);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s : %0d", name, got);
                checks = checks + 1;
            end
        end
    endtask

    // ================================================================== test 1: popcnt9
    reg  [8:0] pv;
    wire [3:0] pcnt;

    popcnt9 u_pc (.din(pv), .cnt(pcnt));

    task test_popcnt9;
        integer i, j, e, nbad;
        begin
            nbad = 0;
            for (i = 0; i < 512; i = i + 1) begin
                pv = i;
                #1;
                e = 0;
                for (j = 0; j < 9; j = j + 1)
                    if (((i >> j) & 1) != 0) e = e + 1;
                if (pcnt !== e[3:0]) begin
                    if (nbad < 4) $display("       popcnt9(%0d) got %0d exp %0d", i, pcnt, e);
                    nbad = nbad + 1;
                end
            end
            if (nbad == 0) begin
                $display("[PASS] popcnt9 exhaustive 0..511 : 512/512");
                checks = checks + 1;
            end else begin
                $display("[FAIL] popcnt9 exhaustive 0..511 : %0d/512 bad", nbad);
                errors = errors + 1;
            end
        end
    endtask

    // ================================================================== test 2: lane_array
    reg  [71:0] la_win, la_wgt;
    wire [31:0] la_cnt;

    lane_array u_la (.win(la_win), .wgt(la_wgt), .cnt(la_cnt));

    task test_lane;
        integer c, t, v, nbad, e;
        reg [71:0] w, g;
        begin
            // --- 2a: Python vector A ---
            la_win = WIN_A; la_wgt = WGT_A; #1;
            if (la_cnt !== CNT_A) begin
                $display("[FAIL] lane_array python vector A : got %08x exp %08x", la_cnt, CNT_A);
                for (c = 0; c < 8; c = c + 1)
                    if (la_cnt[c*4 +: 4] !== CNT_A[c*4 +: 4])
                        $display("       lane %0d got %0d exp %0d", c, la_cnt[c*4 +: 4], CNT_A[c*4 +: 4]);
                errors = errors + 1;
            end else begin
                $display("[PASS] lane_array python vector A : %08x (lanes 3,2,3,3,5,3,5,5)", la_cnt);
                checks = checks + 1;
            end

            // --- 2b: wgt == win -> every lane 9 ---
            la_win = WIN_A; la_wgt = WIN_A; #1;
            if (la_cnt !== 32'h99999999) begin
                $display("[FAIL] lane_array wgt==win : got %08x exp 99999999", la_cnt);
                errors = errors + 1;
            end else begin
                $display("[PASS] lane_array wgt==win : all lanes 9");
                checks = checks + 1;
            end

            // --- 2c: wgt == ~win -> every lane 0 ---
            la_win = WIN_A; la_wgt = ~WIN_A; #1;
            if (la_cnt !== 32'h00000000) begin
                $display("[FAIL] lane_array wgt==~win : got %08x exp 00000000", la_cnt);
                errors = errors + 1;
            end else begin
                $display("[PASS] lane_array wgt==~win : all lanes 0");
                checks = checks + 1;
            end

            // --- 2d: 64 random vectors, TB bit-count reference ---
            nbad = 0;
            for (v = 0; v < 64; v = v + 1) begin
                w = {$random, $random, $random};
                g = {$random, $random, $random};
                la_win = w; la_wgt = g; #1;
                for (c = 0; c < 8; c = c + 1) begin
                    e = 0;
                    for (t = 0; t < 9; t = t + 1)
                        if (w[c*9 + t] == g[c*9 + t]) e = e + 1;
                    if (la_cnt[c*4 +: 4] !== e[3:0]) nbad = nbad + 1;
                end
            end
            if (nbad == 0) begin
                $display("[PASS] lane_array random 64 vectors x 8 lanes : 512/512");
                checks = checks + 1;
            end else begin
                $display("[FAIL] lane_array random 64 vectors x 8 lanes : %0d/512 bad", nbad);
                errors = errors + 1;
            end
        end
    endtask

    // ================================================================== DUT: win_buf W=1
    reg         wb1_sof;
    wire        wb1_we;
    reg  [0:0]  wb1_din;
    wire [8:0]  wb1_win;
    wire        wb1_vld, wb1_req, wb1_done;
    wire [5:0]  wb1_row, wb1_col;

    win_buf #(.W(1), .CW(65), .CR(65)) u_wb1 (
        .clk(clk), .rst_n(rst_n), .sof(wb1_sof), .din_we(wb1_we), .din(wb1_din),
        .win(wb1_win), .win_valid(wb1_vld), .win_row(wb1_row), .win_col(wb1_col),
        .din_req(wb1_req), .done(wb1_done)
    );

    // ================================================================== DUT: win_buf W=8
    reg         wb8_sof;
    wire        wb8_we;
    reg  [7:0]  wb8_din;
    wire [71:0] wb8_win;
    wire        wb8_vld, wb8_req, wb8_done;
    wire [5:0]  wb8_row, wb8_col;

    win_buf #(.W(8), .CW(65), .CR(65)) u_wb8 (
        .clk(clk), .rst_n(rst_n), .sof(wb8_sof), .din_we(wb8_we), .din(wb8_din),
        .win(wb8_win), .win_valid(wb8_vld), .win_row(wb8_row), .win_col(wb8_col),
        .din_req(wb8_req), .done(wb8_done)
    );

    // ================================================================== frame memories
    reg        frm  [0:4095];      // 1-bit frame  (L1 input, FRAME plane)
    reg [7:0]  frm8 [0:4095];      // 8-bit frame  (L2 input, 8 activation planes)

    integer feed_idx, feed_idx8;
    reg     feed_en;

    always @* wb1_din = frm[feed_idx];        // memory word read in a procedural block
    always @* wb8_din = frm8[feed_idx8];

    assign wb1_we = feed_en & wb1_req;        // pixel valid exactly when requested
    assign wb8_we = feed_en & wb8_req;

    always @(posedge clk) begin
        if (rst_n && feed_en && wb1_req) feed_idx  <= feed_idx + 1;
        if (rst_n && feed_en && wb8_req) feed_idx8 <= feed_idx8 + 1;
    end

    // ================================================================== reference models
    // 3x3 window of centre (r,c) from the frame array, zero (bit domain -1) outside.
    // Independent of the shift-register implementation under test.
    function [8:0] ref_win1;
        input integer r;
        input integer c;
        integer ky, kx, rr, cc, v, b;
        begin
            v = 0;
            for (ky = 0; ky < 3; ky = ky + 1) begin
                for (kx = 0; kx < 3; kx = kx + 1) begin
                    rr = r + ky - 1;
                    cc = c + kx - 1;
                    if ((rr >= 0) && (rr < 64) && (cc >= 0) && (cc < 64))
                        b = frm[rr*64 + cc];
                    else
                        b = 0;
                    v = v | (b << (ky*3 + kx));
                end
            end
            ref_win1 = v[8:0];
        end
    endfunction

    function [71:0] ref_win8;
        input integer r;
        input integer c;
        integer ky, kx, rr, cc, ch, b;
        reg [71:0] v;
        reg [7:0]  wd;
        begin
            v = 72'd0;
            for (ch = 0; ch < 8; ch = ch + 1) begin
                for (ky = 0; ky < 3; ky = ky + 1) begin
                    for (kx = 0; kx < 3; kx = kx + 1) begin
                        rr = r + ky - 1;
                        cc = c + kx - 1;
                        if ((rr >= 0) && (rr < 64) && (cc >= 0) && (cc < 64)) begin
                            wd = frm8[rr*64 + cc];
                            b  = (wd >> ch) & 8'h1;
                        end else begin
                            b = 0;
                        end
                        if (b != 0) v[ch*9 + ky*3 + kx] = 1'b1;
                    end
                end
            end
            ref_win8 = v;
        end
    endfunction

    // XNOR match count of channel ch at pixel (r,c) -- == train_bnn10.matches_fixed()
    function integer ref_match;
        input integer r;
        input integer c;
        input integer ch;
        integer ky, kx, rr, cc, b, m;
        begin
            m = 0;
            for (ky = 0; ky < 3; ky = ky + 1) begin
                for (kx = 0; kx < 3; kx = kx + 1) begin
                    rr = r + ky - 1;
                    cc = c + kx - 1;
                    if ((rr >= 0) && (rr < 64) && (cc >= 0) && (cc < 64))
                        b = frm[rr*64 + cc];
                    else
                        b = 0;
                    if (b == ((WGT_L1 >> (ch*9 + ky*3 + kx)) & 1)) m = m + 1;
                end
            end
            ref_match = m;
        end
    endfunction

    function [3:0] ref_theta;
        input integer ch;
        begin
            case (ch)
                0: ref_theta = 4'd4;
                1: ref_theta = 4'd4;
                2: ref_theta = 4'd5;
                3: ref_theta = 4'd3;
                4: ref_theta = 4'd6;
                5: ref_theta = 4'd2;
                6: ref_theta = 4'd4;
                default: ref_theta = 4'd5;
            endcase
        end
    endfunction

    // ================================================================== win_buf monitors
    integer wb1_nbad, wb1_ntot, wb1_sum, wb1_reqs;
    reg     wb1_mon, wb1_spot;
    reg [8:0] wb1_exp;

    always @(posedge clk) begin
        if (rst_n && wb1_mon) begin
            if (wb1_req) wb1_reqs = wb1_reqs + 1;
            if (wb1_vld) begin
                wb1_exp = ref_win1(wb1_row, wb1_col);
                if (wb1_win !== wb1_exp) begin
                    if (wb1_nbad < 5)
                        $display("       wb1 win mismatch (%0d,%0d) got %03x exp %03x",
                                 wb1_row, wb1_col, wb1_win, wb1_exp);
                    wb1_nbad = wb1_nbad + 1;
                end
                wb1_ntot = wb1_ntot + 1;
                wb1_sum  = wb1_sum + wb1_win;
                if (wb1_spot) begin
                    case (wb1_row*64 + wb1_col)
                        0:      chk("wb1/ones pad (0,0)",   wb1_win, 9'h1b0);
                        1:      chk("wb1/ones pad (0,1)",   wb1_win, 9'h1f8);
                        64:     chk("wb1/ones pad (1,0)",   wb1_win, 9'h1b6);
                        65:     chk("wb1/ones pad (1,1)",   wb1_win, 9'h1ff);
                        63:     chk("wb1/ones pad (0,63)",  wb1_win, 9'h0d8);
                        4032:   chk("wb1/ones pad (63,0)",  wb1_win, 9'h036);
                        4095:   chk("wb1/ones pad (63,63)", wb1_win, 9'h01b);
                        default: ;
                    endcase
                end
            end
        end
    end

    integer wb8_nbad, wb8_ntot, wb8_reqs;
    reg     wb8_mon;
    reg [63:0] wb8_sum;
    reg [71:0] wb8_exp;

    always @(posedge clk) begin
        if (rst_n && wb8_mon) begin
            if (wb8_req) wb8_reqs = wb8_reqs + 1;
            if (wb8_vld) begin
                wb8_exp = ref_win8(wb8_row, wb8_col);
                if (wb8_win !== wb8_exp) begin
                    if (wb8_nbad < 5)
                        $display("       wb8 win mismatch (%0d,%0d)", wb8_row, wb8_col);
                    wb8_nbad = wb8_nbad + 1;
                end
                wb8_ntot = wb8_ntot + 1;
                wb8_sum  = wb8_sum + wb8_win;
            end
        end
    end

    // ================================================================== frame builders
    task build_pat1;
        input integer sel;
        integer i, r, c;
        reg [15:0] st;
        begin
            if (sel == 3) begin
                st = 16'hACE1;
                for (i = 0; i < 4096; i = i + 1) begin
                    frm[i] = st[0];
                    st = st[0] ? ((st >> 1) ^ 16'hA001) : (st >> 1);
                end
            end else begin
                for (i = 0; i < 4096; i = i + 1) begin
                    r = i / 64;
                    c = i % 64;
                    case (sel)
                        0:       frm[i] = 1'b1;
                        1:       frm[i] = c & 1;
                        2:       frm[i] = r & 1;
                        default: frm[i] = 1'b0;
                    endcase
                end
            end
        end
    endtask

    task build_pat8;
        integer i;
        reg [15:0] st;
        begin
            st = 16'h1234;
            for (i = 0; i < 4096; i = i + 1) begin
                frm8[i] = st[7:0];
                st = st[0] ? ((st >> 1) ^ 16'hA001) : (st >> 1);
            end
        end
    endtask

    // ================================================================== frame runners
    task run_frame1;
        input [8*32-1:0] name;
        input integer    sel;
        input integer    exp_sum;
        begin
            build_pat1(sel);
            feed_idx = 0;
            wb1_nbad = 0; wb1_ntot = 0; wb1_sum = 0; wb1_reqs = 0;
            @(negedge clk); feed_en = 1'b1; wb1_sof = 1'b1;
            @(negedge clk); wb1_sof = 1'b0;
            wb1_mon  = 1'b1;
            wb1_spot = (sel == 0);
            wait (wb1_done == 1'b1);
            @(negedge clk);              // let the last valid window be sampled
            @(negedge clk);
            wb1_mon = 1'b0; wb1_spot = 1'b0; feed_en = 1'b0;

            if ((wb1_nbad == 0) && (wb1_ntot == 4096) && (wb1_reqs == 4096)) begin
                $display("[PASS] win_buf W1 %0s : 4096/4096 windows, 4096 pixel requests", name);
                checks = checks + 1;
            end else begin
                $display("[FAIL] win_buf W1 %0s : %0d bad / %0d windows / %0d requests",
                         name, wb1_nbad, wb1_ntot, wb1_reqs);
                errors = errors + 1;
            end
            if (wb1_sum === exp_sum) begin
                $display("[PASS] win_buf W1 %0s : window sum %0d (python)", name, wb1_sum);
                checks = checks + 1;
            end else begin
                $display("[FAIL] win_buf W1 %0s : window sum %0d, python %0d", name, wb1_sum, exp_sum);
                errors = errors + 1;
            end
        end
    endtask

    task run_frame8;
        input [8*32-1:0] name;
        input [63:0]     exp_sum;
        begin
            build_pat8;
            feed_idx8 = 0;
            wb8_nbad = 0; wb8_ntot = 0; wb8_reqs = 0; wb8_sum = 64'd0;
            @(negedge clk); feed_en = 1'b1; wb8_sof = 1'b1;
            @(negedge clk); wb8_sof = 1'b0;
            wb8_mon = 1'b1;
            wait (wb8_done == 1'b1);
            @(negedge clk);
            @(negedge clk);
            wb8_mon = 1'b0; feed_en = 1'b0;

            if ((wb8_nbad == 0) && (wb8_ntot == 4096) && (wb8_reqs == 4096)) begin
                $display("[PASS] win_buf W8 %0s : 4096/4096 windows x 8 planes", name);
                checks = checks + 1;
            end else begin
                $display("[FAIL] win_buf W8 %0s : %0d bad / %0d windows / %0d requests",
                         name, wb8_nbad, wb8_ntot, wb8_reqs);
                errors = errors + 1;
            end
            if (wb8_sum === exp_sum) begin
                $display("[PASS] win_buf W8 %0s : window sum 0x%016x (python)", name, wb8_sum);
                checks = checks + 1;
            end else begin
                $display("[FAIL] win_buf W8 %0s : window sum 0x%016x, python 0x%016x",
                         name, wb8_sum, exp_sum);
                errors = errors + 1;
            end
        end
    endtask

    // ================================================================== test 5: accum
    reg         acc_clr, acc_en, acc_ld;
    reg  [2:0]  acc_ld_ch;
    reg  [11:0] acc_ld_data;
    reg  [31:0] acc_din;
    wire [95:0] acc_out;

    accum #(.W(12)) u_acc (
        .clk(clk), .rst_n(rst_n), .clr(acc_clr), .ld(acc_ld), .ld_ch(acc_ld_ch),
        .ld_data(acc_ld_data), .en(acc_en), .din(acc_din), .acc(acc_out)
    );

    task chk_acc(input [8*48-1:0] name, input [95:0] exp);
        begin
            if (acc_out !== exp) begin
                $display("[FAIL] %0s : got %024x exp %024x", name, acc_out, exp);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s : %024x", name, acc_out);
                checks = checks + 1;
            end
        end
    endtask

    task acc_do;                              // one clock cycle with the given controls
        input integer nclr, nen, nld, nch, ndat;
        input [31:0]  ndin;
        begin
            @(negedge clk);
            acc_clr = nclr[0]; acc_en = nen[0]; acc_ld = nld[0];
            acc_ld_ch = nch; acc_ld_data = ndat; acc_din = ndin;
            @(negedge clk);                   // exactly one posedge was applied
            acc_clr = 1'b0; acc_en = 1'b0; acc_ld = 1'b0;
        end
    endtask

    task test_accum;
        begin
            acc_clr = 1'b1; acc_en = 1'b0; acc_ld = 1'b0; acc_ld_ch = 3'd0;
            acc_ld_data = 12'd0; acc_din = 32'd0;
            @(negedge clk); acc_clr = 1'b0;
            @(negedge clk);
            chk_acc("accum reset -> 0", 96'd0);

            // 3 adds of lanes 1,2,3,4,5,6,7,8  (din = 0x87654321)
            acc_do(0, 1, 0, 0, 0, 32'h87654321);
            acc_do(0, 1, 0, 0, 0, 32'h87654321);
            acc_do(0, 1, 0, 0, 0, 32'h87654321);
            chk_acc("accum 3 x en", 96'h01801501200F00C009006003);

            // clr only
            acc_do(1, 0, 0, 0, 0, 32'h87654321);
            chk_acc("accum clr", 96'd0);

            // clr + en -> acc = din
            acc_do(1, 1, 0, 0, 0, 32'h99999999);
            chk_acc("accum clr+en = din", 96'h009009009009009009009009);

            // load channel 5
            acc_do(0, 0, 1, 5, 12'hABC, 32'd0);
            chk_acc("accum ld ch5", 96'h009009ABC009009009009009);

            // one more add of 1
            acc_do(0, 1, 0, 0, 0, 32'h11111111);
            chk_acc("accum ld then en", 96'h00A00AABD00A00A00A00A00A);

            // hold (no en / no clr / no ld)
            acc_do(0, 0, 0, 0, 0, 32'h11111111);
            chk_acc("accum hold", 96'h00A00AABD00A00A00A00A00A);

            acc_clr = 1'b0; acc_en = 1'b0; acc_ld = 1'b0; acc_din = 32'd0;
        end
    endtask

    // ================================================================== test 6: theta_cmp
    reg         th_we;
    reg  [2:0]  th_ch;
    reg  [3:0]  th_data;
    reg  [95:0] acc_in;
    wire [7:0]  acc_act;
    wire [31:0] acc_th;

    theta_cmp #(.W_ACC(12), .W_TH(4)) u_th (
        .clk(clk), .rst_n(rst_n), .th_we(th_we), .th_ch(th_ch), .th_data(th_data),
        .acc(acc_in), .act(acc_act), .th_val(acc_th)
    );

    task th_write;
        input integer ch, v;
        begin
            @(negedge clk);
            th_we = 1'b1; th_ch = ch[2:0]; th_data = v[3:0];
            @(negedge clk);
            th_we = 1'b0;
        end
    endtask

    task th_set;
        input integer ch, v;
        begin
            acc_in[ch*12 +: 12] = v[11:0];
        end
    endtask

    task chk_act(input [8*48-1:0] name, input integer exp);
        integer k;
        begin
            if (acc_act !== exp[7:0]) begin
                $display("[FAIL] %0s : got %02x exp %02x", name, acc_act, exp[7:0]);
                for (k = 0; k < 8; k = k + 1)
                    if (acc_act[k] !== exp[k]) $display("       act[%0d] got %b exp %b", k, acc_act[k], exp[k]);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s : %02x", name, acc_act);
                checks = checks + 1;
            end
        end
    endtask

    task test_theta;
        begin
            acc_in = 96'd0;
            th_write(0, 5); th_write(1, 0); th_write(2, 9); th_write(3, 3);
            th_write(4, 12); th_write(5, 1); th_write(6, 7); th_write(7, 15);
            $display("[info] theta loaded: 5 0 9 3 12 1 7 15");

            // acc == theta -> all 0 (strict >)
            th_set(0, 5); th_set(1, 0); th_set(2, 9); th_set(3, 3);
            th_set(4, 12); th_set(5, 1); th_set(6, 7); th_set(7, 15);
            #1; chk_act("theta_cmp acc == th", 8'h00);

            // acc == theta+1 -> all 1
            th_set(0, 6); th_set(1, 1); th_set(2, 10); th_set(3, 4);
            th_set(4, 13); th_set(5, 2); th_set(6, 8); th_set(7, 16);
            #1; chk_act("theta_cmp acc == th+1", 8'hFF);

            // mixed
            th_set(0, 6); th_set(1, 0); th_set(2, 9); th_set(3, 2);
            th_set(4, 13); th_set(5, 1); th_set(6, 8); th_set(7, 15);
            #1; chk_act("theta_cmp mixed", 8'h51);

            // full-scale 12-bit acc
            th_set(0, 12'hFFF); th_set(1, 12'hFFF); th_set(2, 12'hFFF); th_set(3, 12'hFFF);
            th_set(4, 12'hFFF); th_set(5, 12'hFFF); th_set(6, 12'hFFF); th_set(7, 12'hFFF);
            #1; chk_act("theta_cmp acc full scale", 8'hFF);

            // read-back of the theta registers
            chk("theta read-back ch2", acc_th[2*4 +: 4], 9);
            chk("theta read-back ch7", acc_th[7*4 +: 4], 15);
            acc_in = 96'd0;
        end
    endtask

    // ================================================================== test 7: L1 chain
    wire [71:0] int_win = {8{wb1_win}};       // 1 input plane fanned out to 8 lanes
    wire [31:0] int_cnt;
    wire [95:0] int_acc;
    wire [7:0]  int_act;

    lane_array u_int_la (.win(int_win), .wgt(WGT_L1), .cnt(int_cnt));

    accum #(.W(12)) u_int_acc (
        .clk(clk), .rst_n(rst_n), .clr(wb1_vld), .ld(1'b0), .ld_ch(3'd0),
        .ld_data(12'd0), .en(wb1_vld), .din(int_cnt), .acc(int_acc)
    );

    reg         th_we2;
    reg  [2:0]  th_ch2;
    reg  [3:0]  th_data2;
    wire [31:0] th_val2;

    theta_cmp #(.W_ACC(12), .W_TH(4)) u_int_th (
        .clk(clk), .rst_n(rst_n), .th_we(th_we2), .th_ch(th_ch2), .th_data(th_data2),
        .acc(int_acc), .act(int_act), .th_val(th_val2)
    );

    // one-cycle delayed pixel coordinates: acc/act belong to the window seen last cycle
    reg       int_mon;
    reg       int_vld_d;
    reg [5:0] int_row_d, int_col_d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            int_vld_d <= 1'b0;
            int_row_d <= 6'd0;
            int_col_d <= 6'd0;
        end else begin
            int_vld_d <= wb1_vld & int_mon;
            int_row_d <= wb1_row;
            int_col_d <= wb1_col;
        end
    end

    integer int_nbad_cnt, int_nbad_act, int_ntot, int_sum_cnt, int_act_sum;
    integer int_ch, int_m, int_e;

    always @(posedge clk) begin
        if (rst_n && int_mon && int_vld_d) begin
            for (int_ch = 0; int_ch < 8; int_ch = int_ch + 1) begin
                int_m = ref_match(int_row_d, int_col_d, int_ch);
                int_e = ((int_m > ref_theta(int_ch)) ? 1 : 0);
                if (int_acc[int_ch*12 +: 12] !== int_m[11:0]) begin
                    if (int_nbad_cnt < 5)
                        $display("       int acc ch%0d (%0d,%0d) got %0d exp %0d",
                                 int_ch, int_row_d, int_col_d, int_acc[int_ch*12 +: 12], int_m);
                    int_nbad_cnt = int_nbad_cnt + 1;
                end
                if (int_act[int_ch] !== int_e[0]) begin
                    if (int_nbad_act < 5)
                        $display("       int act ch%0d (%0d,%0d) got %b exp %b",
                                 int_ch, int_row_d, int_col_d, int_act[int_ch], int_e[0]);
                    int_nbad_act = int_nbad_act + 1;
                end
                int_sum_cnt = int_sum_cnt + int_m;
                int_act_sum = int_act_sum + int_e[0];
            end
            int_ntot = int_ntot + 1;
        end
    end

    task test_integrated;
        begin
            // program theta: 4 4 5 3 6 2 4 5
            @(negedge clk);
            th_we2 = 1'b1; th_ch2 = 3'd0; th_data2 = 4'd4; @(negedge clk);
            th_ch2 = 3'd1; th_data2 = 4'd4; @(negedge clk);
            th_ch2 = 3'd2; th_data2 = 4'd5; @(negedge clk);
            th_ch2 = 3'd3; th_data2 = 4'd3; @(negedge clk);
            th_ch2 = 3'd4; th_data2 = 4'd6; @(negedge clk);
            th_ch2 = 3'd5; th_data2 = 4'd2; @(negedge clk);
            th_ch2 = 3'd6; th_data2 = 4'd4; @(negedge clk);
            th_ch2 = 3'd7; th_data2 = 4'd5; @(negedge clk);
            th_we2 = 1'b0;

            build_pat1(3);                     // LFSR frame, same as the python golden
            feed_idx = 0;
            int_nbad_cnt = 0; int_nbad_act = 0; int_ntot = 0;
            int_sum_cnt = 0;  int_act_sum = 0;
            @(negedge clk); feed_en = 1'b1; wb1_sof = 1'b1; int_mon = 1'b1;
            @(negedge clk); wb1_sof = 1'b0;
            wait (wb1_done == 1'b1);
            @(negedge clk);
            @(negedge clk);
            @(negedge clk);              // the delayed-valid monitor samples 2 cycles later
            int_mon = 1'b0; feed_en = 1'b0;

            if ((int_nbad_cnt == 0) && (int_ntot == 4096)) begin
                $display("[PASS] L1 chain acc : 4096 pixels x 8 ch vs reference model");
                checks = checks + 1;
            end else begin
                $display("[FAIL] L1 chain acc : %0d bad of %0d pixels (8 ch each)",
                         int_nbad_cnt, int_ntot);
                errors = errors + 1;
            end
            if ((int_nbad_act == 0) && (int_ntot == 4096)) begin
                $display("[PASS] L1 chain act : 4096 pixels x 8 ch vs reference model");
                checks = checks + 1;
            end else begin
                $display("[FAIL] L1 chain act : %0d bad of %0d pixels", int_nbad_act, int_ntot);
                errors = errors + 1;
            end

            // python aggregates over the whole activation plane (independent script)
            chk("L1 chain count sum  (python 147354)", int_sum_cnt, 147354);
            chk("L1 chain act ones   (python 15326)",  int_act_sum, 15326);
        end
    endtask

    // python spot value for pixel (31,17): counts [3,4,4,6,5,4,5,9], act 0xE8
    always @(posedge clk) begin
        if (rst_n && int_mon && int_vld_d && (int_row_d == 6'd31) && (int_col_d == 6'd17)) begin
            if (int_acc === 96'h009005004005006004004003) begin
                $display("[PASS] L1 chain spot (31,17) counts [3,4,4,6,5,4,5,9] (python)");
                checks = checks + 1;
            end else begin
                $display("[FAIL] L1 chain spot (31,17) counts got %024x", int_acc);
                errors = errors + 1;
            end
            if (int_act === 8'hE8) begin
                $display("[PASS] L1 chain spot (31,17) act 0xE8 (python)");
                checks = checks + 1;
            end else begin
                $display("[FAIL] L1 chain spot (31,17) act got %02x", int_act);
                errors = errors + 1;
            end
        end
    end

    // ================================================================== main
    initial begin
        errors = 0; checks = 0;

        wb1_sof = 1'b0; wb1_din = 1'b0; feed_en = 1'b0; feed_idx = 0; feed_idx8 = 0;
        wb8_sof = 1'b0; wb8_din = 8'd0;
        wb1_mon = 1'b0; wb1_spot = 1'b0; wb8_mon = 1'b0; int_mon = 1'b0;
        wb1_nbad = 0; wb1_ntot = 0; wb1_sum = 0; wb1_reqs = 0;
        wb8_nbad = 0; wb8_ntot = 0; wb8_reqs = 0; wb8_sum = 64'd0;
        acc_clr = 1'b0; acc_en = 1'b0; acc_ld = 1'b0; acc_ld_ch = 3'd0;
        acc_ld_data = 12'd0; acc_din = 32'd0;
        th_we = 1'b0; th_ch = 3'd0; th_data = 4'd0; acc_in = 96'd0;
        th_we2 = 1'b0; th_ch2 = 3'd0; th_data2 = 4'd0;

        rst_n = 1'b0;
        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        repeat (3) @(negedge clk);

        $display("[tb] ===== BNN engine skeleton (popcnt9 / lane_array / accum / theta_cmp / win_buf) =====");

        test_popcnt9;
        test_lane;
        run_frame1("all-ones",  0, 2040901);
        run_frame1("col-parity", 1, 1027361);
        run_frame1("row-parity", 2, 1034405);
        run_frame1("lfsr",      3, 1024051);
        run_frame8("lfsr-8plane", 64'hc596422fc6118dda);
        test_accum;
        test_theta;
        test_integrated;

        if (errors == 0) $display("[RESULT] ALL PASS (%0d checks)", checks);
        else             $display("[RESULT] %0d FAILURE(S) (%0d checks)", errors, checks);
        $finish;
    end

    // global timeout
    initial begin
        #5000000;
        $display("[FAIL] global timeout (5 ms)");
        $finish;
    end

endmodule
