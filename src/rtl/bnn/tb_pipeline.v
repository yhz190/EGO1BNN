// tb_pipeline.v -- self-checking testbench for the v2.2-patch2 datapath
//                  (act_bram / layer_seq / l3_mix 8ch / l4_pool 8ch / l5_fc 2048 Q6
//                   + the verified engine core win_buf / lane_array / accum / theta_cmp)
//
// patch2 (2026-11-01): L3 4 -> 8 channels, L4 4 -> 8 channels (4x4 pool, >=8/16),
// L5 1024 -> 2048 features (256 beats) with a 16 bit signed Q6 bias and a 19 bit
// signed score.  L5 = 257 cycles, whole frame = 16,911 cycles.
//
//   test 1 : win_buf W=8, L1 mode  din = {8{frame_bit}}  -> 4096 windows vs array model
//   test 2 : win_buf W=8, L2 mode  din = 8 planes        -> 4096 windows vs array model
//   test 3 : layer_seq timing: S_L1/S_L2 = 4,228 cycles, S_L3/S_L4 = 4,098,
//            S_L5 = 257, frame = 16,911; beats, barrier, din_req protocol
//   test 4 : l3_mix  8 channels: 2 Python vectors + 2 directed + 16 random vs popcount8
//   test 5 : l4_pool 8 channels: majority / threshold / 64 bit packing / channel readback
//   test 6 : l5_fc   2048 -> 4: 256 beats, Q6 bias (16 bit signed), score (19 bit signed),
//            E6 near tie (0.95 vs 0.97) and both range corners
//   test 7 : full L1..L5 frame: every layer bit-exact against (a) the TB array model
//            and (b) the Python integer aggregates of py/ref_model_patch2.py
//
// Per-beat datapath monitors (patch2 additions, they run on the e2e frame):
//   * L3: every pixel's 8 match counts (l3_cnt, 32 bit) vs an independent popcount8
//   * L4: per-block per-channel sums rebuilt from the pixel stream vs l4_feat
//   * L5: every beat's feat8 vs ref_feat (channel/block mapping) and a TB-side
//         accumulator that must equal l5_fc's acc at the valid pulse
//
// Python goldens (W1/W2/W3/WF0..3/EXP_*) come from
// build_sim_bnn/patch2/py/ref_model_patch2.py -- pure integer, no numpy/torch,
// implements train_bnn10.hw_eval_q6() with pool=4.
//
// Run (see 04_娴嬭瘯/20261101_patch2_RTL閫傞厤璁板綍.md for the recorded commands):
//   xvlog -i rtl popcnt9.v lane_array.v accum.v theta_cmp.v win_buf.v act_bram.v \
//         l3_mix.v l4_pool.v l5_fc.v layer_seq.v tb_pipeline.v
//   xelab tb_pipeline -s sim -i rtl
//   xsim sim -runall
`timescale 1ns / 1ps

module tb_pipeline;

    // ================================================================ clock / reset
    reg clk, rst_n;
    initial clk = 1'b0;
    always #5 clk = ~clk;                          // 100 MHz TB clock (timing is not modelled)

    integer errors, checks;

    task chk(input [8*60-1:0] name, input integer got, input integer exp);
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

    task chk32(input [8*60-1:0] name, input [31:0] got, input [31:0] exp);
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

    task chk64(input [8*60-1:0] name, input [63:0] got, input [63:0] exp);
        begin
            if (got !== exp) begin
                $display("[FAIL] %0s : got %016x, exp %016x", name, got, exp);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s : %016x", name, got);
                checks = checks + 1;
            end
        end
    endtask

    // ================================================================ Python goldens
    // ref_model_patch2.py: 16-bit LFSR seeds 0x1001/0x1002/0x1003/0x1004
    localparam [71:0]   W1 = 72'hb00e40170034004fff;
    localparam [71:0]   W2 = 72'h701f402b0064008ffe;
    localparam [63:0]   W3 = 64'hefbfc0ffabff3001;
    localparam [2047:0] WF0 = 2048'h38145fcf2aeb9965dc8d2e1397c5caf2e6b9765cb72e4b971a3427b075bf4dab1264388fa1ea7d675374c4b10e43e8fa9ea62988621f782b5f9b2a24678f75eb4d64ec8e91e9c29d062c086fe0957e3357bb35a44d8f121438305fbf2a5467308bbe1a57d8cadee6d689361c482f1f942a3067bf75ab4d9b122438705f40d4fecea91663c8851e0c2810603f7fab559b3224478f0a14183020407f00abfe65577334bbb1a5bd8daded929239c7a2f586b209b81da02d806dff6d549331c442f0f9415cfcd1513cc3aefa69589cde2ed796ca36e7b68a49e71d742cb06e409701cbfd1aac266f8895e1cd7d1353c4c50ef3e945630c8411f03d405300c4010ffc;
    localparam [2047:0] WF1 = 2048'h32bc4650f73eb4564f37144bcf1aebd965237384ba0e581720347fb0aa4198fddead29939dc5d2f2c6b909a3e27a8759f4dd4ed316c436f04940e3017bfca5518cc21107c3f5054c0cefee9569cc9d11d3c2c506f3f6bab659b7224b871a0bd81adfd92adc66d0893e1c57d0cac11903dc052ff39545ccf2eeb969a3627b78a4a18e7de8ad61937dc4ad0e6c176fcb6ae49971dcbd2e53973a34584f20eb8165fc8d51ecc291063c08501f3fd45530cc4110fc3eafa9959dcdd2ed396c5c90d1c13d03ac05900dc012ffc6aaf666b7764b48e4e1717cbcae51973dcbad1a6c276f8b6a1b67db7524b38e45e8f29eb9d65d372c4b90e5c172fcb951a33d84520f;
    localparam [2047:0] WF2 = 2048'haeb669b7624b78e4a171834204f80ea0167fc8aae1997ddcad2e6c68909e3e28579f35d44d30ec416f036bfb655b7324bb8e5a1727cb8ae5e68d7613483b1fa42a7067408b01e4028ff9eaa299862208781f5fd4d53133bc45af0d94123038405f00d4013003bffa5558ccdeeed696c9c91d1c2c2f906a3f67ab759b4ddb12dbc6daf6d949231c7bd0a53e73a8ba61a77d8b521b38245f8f2a1467cf7514b3ce45170c34104fc0eafe99562337844a0f181420307fbf55ab32644770f4beb1a9bd9dadd26d3893a1c582f206b809a01d802dff92aa39985ddf2d2b939a3a27a78a75e74d74ecb16e4368fb615b7cdb5124c38efa16a7c98ae21978235f84d5f1;
    localparam [2047:0] WF3 = 2048'h37e84a9f19d422cf86ea0967e3757b4ca4ee7168bc9e51d73d3453b0c5bef256b8c9a11d83d205380c5fef2a9466308841e0fd7eaca9919dc22d0793f5c54d0cec116fc36afb66a4898e1de82d606c8091fe3d57acca6ee769749cb1d1bd3dac526f38945e30d7beca56e7368bb61a48271f8bd5e5328c4610f7c14afce6ae89961dc82d1f93d5c532f3b945a30d84120fc7eaf566b37644b70e4be8e561737cbb51a4c2710743f4fab159bcddaed269389c5e2f286b9f65d48d31ec4290f9c15d032c046ff09541ccfd1153c33afba6a5898de21287c60af7e6b5764cb711b43db05240c7010bfc1aafd995223387ba0a5818dfded529339c45d0f2c146fcf6;

    localparam integer  EXP_SUM1 = 13454;
    localparam integer  EXP_SUM2 = 18629;
    localparam integer  EXP_SUM3 = 14751;
    localparam integer  EXP_SUM4 = 995;
    localparam [31:0]   EXP_H1 = 32'hb245f404;
    localparam [31:0]   EXP_H2 = 32'h0357acbe;
    localparam [31:0]   EXP_H3 = 32'h95e68a3c;
    localparam [31:0]   EXP_H4 = 32'h247e7251;

    // L5 e2e result (Q6): acc 999/1015/1010/1022, score -3202/-1155/-1794/-255
    localparam integer  EXP_ACC0 = 999, EXP_ACC1 = 1015, EXP_ACC2 = 1010, EXP_ACC3 = 1022;
    localparam integer  EXP_SC0 = -3202, EXP_SC1 = -1155, EXP_SC2 = -1794, EXP_SC3 = -255;

    // l3_mix Python vectors (ref_model_patch2.py "l3 case 0..3")
    localparam [63:0]   L3V2_W = 64'h3c69a5f0deadbeef;
    localparam [7:0]    L3V2_A = 8'b0000_1010;
    localparam [63:0]   L3V3_W = 64'ha55a3cc3f00f0ff0;
    localparam [31:0]   L3V3_C = 32'h4444_4444;
    localparam [7:0]    L3V3_A = 8'b1010_1010;

    // layer_seq state encoding (mirrors the localparams in layer_seq.v)
    localparam [2:0] S_IDLE = 3'd0, S_L1 = 3'd1, S_L1_DONE = 3'd2, S_L2 = 3'd3,
                     S_L3 = 3'd4, S_L4 = 3'd5, S_L5 = 3'd6, S_DONE = 3'd7;

    // expected layer durations (state cycles) and total frame latency -- patch2 鍙ｅ緞
    localparam integer EXP_L1_CYC = 4228, EXP_L2_CYC = 4228, EXP_L3_CYC = 4098,
                       EXP_L4_CYC = 4098, EXP_L5_CYC = 257, EXP_FRAME_CYC = 16911;

    // ================================================================ stimulus data
    reg  frm  [0:4095];                 // test image, bit=1 -> +1
    reg  [3:0]  TH1  [0:7];
    reg  [3:0]  TH2  [0:7];
    reg  [3:0]  TH3  [0:7];
    reg  [4:0]  TH4  [0:7];
    reg  signed [15:0] THFC [0:3];      // Q6
    reg  [2047:0] WFA [0:3];            // WF0..WF3 for variable indexing

    // reference activations (computed by the array model *before* the DUT runs)
    reg  [7:0]  ref_a1 [0:4095];
    reg  [7:0]  ref_a2 [0:4095];
    reg  [7:0]  ref_a3 [0:4095];
    reg  [7:0]  ref_feat [0:255];
    integer     ref_acc  [0:3];
    integer     ref_score[0:3];

    // ================================================================ DUT wires
    reg  seq_start;

    wire        busy, layer_done, all_done, layer_start, sof, din_we;
    wire        rd_en, rd_bank, wr_en, wr_bank, wr_lo4, pix_en;
    wire [2:0]  layer, done_layer, state;
    wire [11:0] pix_idx, rd_addr, waddr;
    wire [12:0] beat_cnt, win_cnt, cyc_cnt, layer_beats;

    wire [71:0] wb_win;
    wire        wb_vld, wb_done, wb_req;
    wire [5:0]  wb_row, wb_col;

    wire [31:0] lane_cnt;
    wire [95:0] acc;
    wire [7:0]  act1, act2, act8;
    wire [7:0]  act_rdata;

    wire [7:0]  l3_act;                 // patch2: 8 output channels
    wire [31:0] l3_cnt;                 // patch2: 8 x 4 bit counts
    wire        l3_valid;

    wire [7:0]  l4_feat;                // patch2: 8 channels
    wire        l4_feat_we;
    wire [7:0]  l4_feat_blk;
    wire [7:0]  l4_frd_data;

    wire        l5_valid;
    wire [75:0] l5_score;               // patch2: 4 x 19 bit signed
    wire [47:0] l5_acc;                 // patch2: 4 x 12 bit
    wire [4:0]  l5_fa;
    wire [2:0]  l5_fch;

    // ---- test-mode muxes (dir_mode = 1 during the directed module tests) ----
    reg         dir_mode;
    reg         tb_l3_en, tb_l4_en, tb_l4_sof, tb_fc_start;
    reg  [7:0]  tb_l3_din, tb_fc_feat;
    reg  [63:0] tb_w3;
    reg  [31:0] tb_th3;
    reg  [7:0]  tb_l4_din;
    reg  [39:0] tb_th4;
    reg  [4:0]  tb_frd_addr;
    reg  [2:0]  tb_frd_ch;

    wire        l3_en   = dir_mode ? tb_l3_en : (pix_en && (layer == 3'd3));
    wire [7:0]  l3_din  = dir_mode ? tb_l3_din : act_rdata;
    wire        l4_en   = dir_mode ? tb_l4_en : (pix_en && (layer == 3'd4));
    wire [7:0]  l4_din  = dir_mode ? tb_l4_din : act_rdata;
    wire        l4_sof  = dir_mode ? tb_l4_sof : (layer_start && (layer == 3'd4));
    wire        fc_start= dir_mode ? tb_fc_start : (layer_start && (layer == 3'd5));
    wire [7:0]  fc_feat = dir_mode ? tb_fc_feat : l4_frd_data;
    wire [4:0]  frd_addr= dir_mode ? tb_frd_addr : l5_fa;
    wire [2:0]  frd_ch  = dir_mode ? tb_frd_ch : l5_fch;

    // ---- thetas / weight loads ----
    reg         th_we, th_sel;
    reg  [2:0]  th_ch;
    reg  [3:0]  th_data;
    wire        th_we1 = th_we && !th_sel;
    wire        th_we2 = th_we &&  th_sel;
    wire [31:0] th_val1, th_val2;

    reg         wf_we;
    reg  [7:0]  wf_addr;                // patch2: 256 weight words
    reg  [31:0] wf_data;
    reg         tf_we;
    reg  [1:0]  tf_ch;
    reg  [15:0] tf_data;                // patch2: Q6, 16 bit signed

    wire [71:0] lane_wgt = (layer == 3'd1) ? W1 : W2;
    wire [7:0]  act_wdata = wr_lo4 ? l3_act : act8;
    wire [7:0]  wb_din = (layer == 3'd1) ? {8{frm[pix_idx]}} : act_rdata;

    // ================================================================ DUT instances
    // NOTE: no parameter override on purpose -- the TB runs the DUT *defaults*
    // (N_SCAN = 4225, N_PIX = 4096, N_FC = 256) so that any drift of the default
    // beat counts is caught by the timing checks below (mutation m4 proved this
    // matters: with an explicit .N_FC(256) a wrong default went undetected).
    layer_seq u_seq (
        .clk(clk), .rst_n(rst_n), .start(seq_start), .win_valid(wb_vld),
        .busy(busy), .layer(layer), .layer_done(layer_done), .done_layer(done_layer),
        .all_done(all_done), .state(state), .layer_start(layer_start),
        .sof(sof), .din_we(din_we), .pix_idx(pix_idx),
        .rd_en(rd_en), .rd_bank(rd_bank), .rd_addr(rd_addr),
        .wr_en(wr_en), .wr_bank(wr_bank), .wr_lo4(wr_lo4), .waddr(waddr),
        .pix_en(pix_en), .beat_cnt(beat_cnt), .win_cnt(win_cnt),
        .cyc_cnt(cyc_cnt), .layer_beats(layer_beats)
    );

    win_buf #(.W(8)) u_wb (
        .clk(clk), .rst_n(rst_n), .sof(sof), .din_we(din_we), .din(wb_din),
        .win(wb_win), .win_valid(wb_vld), .win_row(wb_row), .win_col(wb_col),
        .din_req(wb_req), .done(wb_done)
    );

    lane_array u_la (.win(wb_win), .wgt(lane_wgt), .cnt(lane_cnt));

    accum #(.W(12)) u_acc (
        .clk(clk), .rst_n(rst_n), .clr(wb_vld), .ld(1'b0), .ld_ch(3'd0),
        .ld_data(12'd0), .en(wb_vld), .din(lane_cnt), .acc(acc)
    );

    theta_cmp #(.W_ACC(12), .W_TH(4)) u_th1 (
        .clk(clk), .rst_n(rst_n), .th_we(th_we1), .th_ch(th_ch), .th_data(th_data),
        .acc(acc), .act(act1), .th_val(th_val1)
    );

    theta_cmp #(.W_ACC(12), .W_TH(4)) u_th2 (
        .clk(clk), .rst_n(rst_n), .th_we(th_we2), .th_ch(th_ch), .th_data(th_data),
        .acc(acc), .act(act2), .th_val(th_val2)
    );

    assign act8 = (layer == 3'd1) ? act1 : act2;

    act_bram #(.P(8), .AW(12), .DEPTH(4096)) u_act (
        .clk(clk), .we(wr_en), .wbank(wr_bank), .waddr(waddr), .wdata(act_wdata),
        .re(rd_en), .rbank(rd_bank), .raddr(rd_addr), .rdata(act_rdata)
    );

    l3_mix #(.COUT(8), .W_TH(4)) u_l3 (
        .clk(clk), .rst_n(rst_n), .en(l3_en), .din(l3_din), .w3(tb_w3), .th3(tb_th3),
        .act(l3_act), .cnt(l3_cnt), .valid(l3_valid)
    );

    l4_pool #(.NCH(8), .NF(256), .W_TH(5)) u_l4 (
        .clk(clk), .rst_n(rst_n), .sof(l4_sof), .en(l4_en), .din(l4_din),
        .th4(tb_th4), .feat_we(l4_feat_we), .feat_blk(l4_feat_blk), .feat(l4_feat),
        .frd_addr(frd_addr), .frd_ch(frd_ch), .frd_data(l4_frd_data)
    );

    l5_fc #(.NF(2048), .NC(4), .NCH(8), .W_TH(16), .SW(19)) u_l5 (
        .clk(clk), .rst_n(rst_n), .start(fc_start), .feat8(fc_feat),
        .feat_addr(l5_fa), .feat_ch(l5_fch),
        .w_we(wf_we), .w_addr(wf_addr), .w_data(wf_data),
        .th_we(tf_we), .th_ch(tf_ch), .th_data(tf_data),
        .valid(l5_valid), .score(l5_score), .acc_out(l5_acc)
    );

    // ================================================================ helpers
    function integer pc8;                       // independent popcount8
        input [7:0] x;
        integer i;
        begin
            pc8 = 0;
            for (i = 0; i < 8; i = i + 1) pc8 = pc8 + ((x >> i) & 1);
        end
    endfunction

    function integer pc16;
        input [15:0] x;
        integer i;
        begin
            pc16 = 0;
            for (i = 0; i < 16; i = i + 1) pc16 = pc16 + ((x >> i) & 1);
        end
    endfunction

    function integer rwin9;                     // 3x3 window value at centre (r,c)
        input integer r, c;
        integer ky, kx, rr, cc, v;
        begin
            v = 0;
            for (ky = 0; ky < 3; ky = ky + 1)
                for (kx = 0; kx < 3; kx = kx + 1) begin
                    rr = r + ky - 1;
                    cc = c + kx - 1;
                    if ((rr >= 0) && (rr < 64) && (cc >= 0) && (cc < 64) && (frm[rr*64+cc] != 0))
                        v = v | (1 << (ky*3 + kx));
                end
            rwin9 = v;
        end
    endfunction

    function [71:0] rwin8_l2;                   // 8-plane window from the reference a1
        input integer r, c;
        integer ch, ky, kx, rr, cc, b;
        reg [71:0] v;
        begin
            v = 72'd0;
            for (ch = 0; ch < 8; ch = ch + 1)
                for (ky = 0; ky < 3; ky = ky + 1)
                    for (kx = 0; kx < 3; kx = kx + 1) begin
                        rr = r + ky - 1;
                        cc = c + kx - 1;
                        b = 0;
                        if ((rr >= 0) && (rr < 64) && (cc >= 0) && (cc < 64))
                            b = (ref_a1[rr*64+cc] >> ch) & 1;
                        if (b != 0) v[ch*9 + ky*3 + kx] = 1'b1;
                    end
            rwin8_l2 = v;
        end
    endfunction

    function [7:0] l4_ref;                      // majority, pat = 8 ch x 16 bit
        input [127:0] pat;
        input [39:0]  th;
        integer ch, s;
        begin
            for (ch = 0; ch < 8; ch = ch + 1) begin
                s = pc16(pat[ch*16 +: 16]);
                l4_ref[ch] = (s >= ((th >> (ch*5)) & 40'h1F));
            end
        end
    endfunction

    // TH3[]/TH4[] packed into the bus format of the DUT ports (set by build_thv)
    reg [31:0] TH3V;
    reg [39:0] TH4V;
    task build_thv;
        integer ch;
        begin
            for (ch = 0; ch < 8; ch = ch + 1) begin
                TH3V[ch*4 +: 4] = TH3[ch];
                TH4V[ch*5 +: 5] = TH4[ch];
            end
        end
    endtask

    // ================================================================ frame / reference
    task build_frame;
        integer p, r, c, v;
        begin
            for (p = 0; p < 4096; p = p + 1) begin
                r = p / 64;
                c = p % 64;
                v = 0;
                if (((r-20)*(r-20) + (c-20)*(c-20)) < 400)         v = 1;   // disc r=20
                if ((r >= 40) && (r <= 55) && (c >= 6) && (c <= 58)) v = 1; // bar
                if (((r + c) % 17) == 0)                           v = 1;   // stripes
                frm[p] = (v != 0);
            end
        end
    endtask

    task build_ref;
        integer p, r, c, ch, k, ky, kx, rr, cc, b, s, by, bx, B, ry, rx, f, fb;
        begin
            for (k = 0; k < 4; k = k + 1) WFA[k] = (k == 0) ? WF0 :
                                                  (k == 1) ? WF1 :
                                                  (k == 2) ? WF2 : WF3;
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
                            if ((rr >= 0) && (rr < 64) && (cc >= 0) && (cc < 64)) b = frm[rr*64+cc];
                            if (b == ((W1 >> (ch*9 + ky*3 + kx)) & 1)) s = s + 1;
                        end
                    ref_a1[p][ch] = (s > TH1[ch]);
                end
            end
            // ---------------- L2 ----------------
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
                                b = (ref_a1[rr*64+cc] >> ch) & 1;
                            if (b == ((W2 >> (ch*9 + ky*3 + kx)) & 1)) s = s + 1;
                        end
                    ref_a2[p][ch] = (s > TH2[ch]);
                end
            end
            // ---------------- L3 (8 out channels) ----------------
            for (p = 0; p < 4096; p = p + 1) begin
                for (k = 0; k < 8; k = k + 1) begin
                    s = 0;
                    for (ch = 0; ch < 8; ch = ch + 1)
                        if (ref_a2[p][ch] == ((W3 >> (k*8 + ch)) & 1)) s = s + 1;
                    ref_a3[p][k] = (s > TH3[k]);
                end
            end
            // ---------------- L4 (8 channels, 4x4 blocks) ----------------
            for (by = 0; by < 16; by = by + 1)
                for (bx = 0; bx < 16; bx = bx + 1) begin
                    B = by*16 + bx;
                    for (ch = 0; ch < 8; ch = ch + 1) begin
                        s = 0;
                        for (ry = 0; ry < 4; ry = ry + 1)
                            for (rx = 0; rx < 4; rx = rx + 1)
                                s = s + ref_a3[(4*by+ry)*64 + (4*bx+rx)][ch];
                        ref_feat[B][ch] = (s >= TH4[ch]);
                    end
                end
            // ---------------- L5 (2048 -> 4, Q6 score) ----------------
            for (k = 0; k < 4; k = k + 1) begin
                s = 0;
                for (f = 0; f < 2048; f = f + 1) begin
                    fb = ref_feat[f % 256][f / 256];
                    if (fb == WFA[k][f]) s = s + 1;
                end
                ref_acc[k]   = s;
                ref_score[k] = (2*s - 2048) * 64 + THFC[k];
            end
        end
    endtask

    task print_ref_summary;
        integer p, ch, s;
        begin
            s = 0; for (p = 0; p < 4096; p = p + 1) for (ch = 0; ch < 8; ch = ch + 1) s = s + ref_a1[p][ch];
            $display("[ref] L1 act ones   = %0d", s);
            s = 0; for (p = 0; p < 4096; p = p + 1) for (ch = 0; ch < 8; ch = ch + 1) s = s + ref_a2[p][ch];
            $display("[ref] L2 act ones   = %0d", s);
            s = 0; for (p = 0; p < 4096; p = p + 1) for (ch = 0; ch < 8; ch = ch + 1) s = s + ref_a3[p][ch];
            $display("[ref] L3 act ones   = %0d (8 channels)", s);
            s = 0; for (p = 0; p < 256; p = p + 1) for (ch = 0; ch < 8; ch = ch + 1) s = s + ref_feat[p][ch];
            $display("[ref] L4 feat ones  = %0d (2048 bit)", s);
            $display("[ref] L5 acc        = %0d %0d %0d %0d", ref_acc[0], ref_acc[1], ref_acc[2], ref_acc[3]);
            $display("[ref] L5 score Q6   = %0d %0d %0d %0d",
                     ref_score[0], ref_score[1], ref_score[2], ref_score[3]);
        end
    endtask

    // ================================================================ monitors
    integer mon_en;
    integer nbad_win1, nbad_win2, nwin1, nwin2;
    integer nbad_a1, nbad_a2, nbad_a3, nbad_feat, nbad_sc, nbad_acc;
    integer nw1, nw2, nw3, nfeat, nvalid5;
    integer sum1, sum2, sum3, sum4;
    integer ndin1, ndin2, nreq1, nreq2, nproto_bad, nbuf_done;
    integer n_st [0:7];
    integer n_done, n_bad_order, n_barrier;
    integer abs_cyc, t_start, t_done, t_l1_wr_last, t_l2_rd_first;
    integer frame_cyc;
    integer lb_seen [0:4];
    reg [31:0] h1, h2, h3, h4;
    reg [8:0]  exp9;
    integer kk, cc2;

    // ---- patch2 per-beat monitors ----
    integer nbad_l3c, nchk_l3c;            // L3 counts per pixel
    integer nbad_l4s, nchk_l4s;            // L4 per-block sums
    integer nbad_l5b, nchk_l5b;            // L5 feat8 mapping per beat
    integer tb_acc [0:3];                  // TB-side L5 accumulator
    reg [31:0] pexp;
    reg        pexp_v;
    integer l4sum [0:7], l4pc;
    reg [7:0]  l4e_feat;
    reg        l4e_v;
    reg [7:0]  l5e_feat;
    integer    l5q;

    always @(posedge clk) abs_cyc = abs_cyc + 1;

    // ---- win_buf window checks (L1 mode / L2 mode) ----
    always @(posedge clk) begin
        if (rst_n && (mon_en != 0) && wb_vld) begin
            if (layer == 3'd1) begin
                nwin1 = nwin1 + 1;
                exp9  = rwin9(wb_row, wb_col);
                if (wb_win !== {8{exp9}}) begin
                    if (nbad_win1 < 5)
                        $display("       L1 win (%0d,%0d) got %018x exp %018x",
                                 wb_row, wb_col, wb_win, {8{exp9}});
                    nbad_win1 = nbad_win1 + 1;
                end
            end else if (layer == 3'd2) begin
                nwin2 = nwin2 + 1;
                if (wb_win !== rwin8_l2(wb_row, wb_col)) begin
                    if (nbad_win2 < 5)
                        $display("       L2 win (%0d,%0d) got %018x", wb_row, wb_col, wb_win);
                    nbad_win2 = nbad_win2 + 1;
                end
            end
        end
    end

    // ---- ACT write stream checks + Python hashes / sums ----
    always @(posedge clk) begin
        if (rst_n && (mon_en != 0) && wr_en) begin
            if (wr_lo4) begin                                    // L3
                nw3  = nw3 + 1;
                sum3 = sum3 + pc8(act_wdata);
                h3   <= h3 * 33 + {24'd0, act_wdata};
                if (act_wdata !== ref_a3[waddr]) begin
                    if (nbad_a3 < 5)
                        $display("       L3 write p=%0d got %02x exp %02x", waddr, act_wdata, ref_a3[waddr]);
                    nbad_a3 = nbad_a3 + 1;
                end
            end else if (wr_bank) begin                          // L2
                nw2  = nw2 + 1;
                sum2 = sum2 + pc8(act_wdata);
                h2   <= h2 * 33 + {24'd0, act_wdata};
                if (act_wdata !== ref_a2[waddr]) begin
                    if (nbad_a2 < 5)
                        $display("       L2 write p=%0d got %02x exp %02x", waddr, act_wdata, ref_a2[waddr]);
                    nbad_a2 = nbad_a2 + 1;
                end
            end else begin                                       // L1
                nw1  = nw1 + 1;
                sum1 = sum1 + pc8(act_wdata);
                h1   <= h1 * 33 + {24'd0, act_wdata};
                if (act_wdata !== ref_a1[waddr]) begin
                    if (nbad_a1 < 5)
                        $display("       L1 write p=%0d got %02x exp %02x", waddr, act_wdata, ref_a1[waddr]);
                    nbad_a1 = nbad_a1 + 1;
                end
            end
        end
    end

    // ---- L3 per-beat: all 8 match counts of every pixel vs an independent popcount8 ----
    // stage 1 captures the expectation of the pixel consumed this cycle (l3_mix registers
    // its output, so the comparison happens one cycle later in stage 2).
    always @(posedge clk) begin
        if (rst_n && (mon_en != 0) && !dir_mode && (layer == 3'd3) && pix_en) begin
            for (cc2 = 0; cc2 < 8; cc2 = cc2 + 1)
                pexp[cc2*4 +: 4] <= pc8(~(act_rdata ^ W3[cc2*8 +: 8]));
            pexp_v <= 1'b1;
        end else begin
            pexp_v <= 1'b0;
        end
    end

    always @(posedge clk) begin
        if (rst_n && (mon_en != 0) && !dir_mode && pexp_v && (layer == 3'd3)) begin
            nchk_l3c = nchk_l3c + 1;
            if ((l3_valid !== 1'b1) || (l3_cnt !== pexp)) begin
                if (nbad_l3c < 5)
                    $display("       L3 cnt beat got %08x exp %08x (valid %0b)", l3_cnt, pexp, l3_valid);
                nbad_l3c = nbad_l3c + 1;
            end
        end
    end

    // ---- L4 per-block: rebuild the 8 channel sums from the pixel stream ----
    always @(posedge clk) begin
        if (rst_n && (mon_en != 0) && !dir_mode) begin
            if ((layer == 3'd4) && pix_en) begin
                for (cc2 = 0; cc2 < 8; cc2 = cc2 + 1)
                    l4sum[cc2] = l4sum[cc2] + ((act_rdata >> cc2) & 1);
                if (l4pc == 15) begin
                    for (cc2 = 0; cc2 < 8; cc2 = cc2 + 1)
                        l4e_feat[cc2] = (l4sum[cc2] >= TH4[cc2]);
                    l4e_v = 1;
                    l4pc  = 0;
                    for (cc2 = 0; cc2 < 8; cc2 = cc2 + 1) l4sum[cc2] = 0;
                end else begin
                    l4pc = l4pc + 1;
                end
            end
            if (l4_feat_we) begin
                nchk_l4s = nchk_l4s + 1;
                if (l4e_v !== 1'b1 || l4_feat !== l4e_feat) begin
                    if (nbad_l4s < 5)
                        $display("       L4 blk sum got %02x exp %02x (blk %0d)",
                                 l4_feat, l4e_feat, l4_feat_blk);
                    nbad_l4s = nbad_l4s + 1;
                end
                l4e_v = 0;
            end
        end
    end

    // ---- L4 feature stream: every block vs the reference + Python hash/sum ----
    always @(posedge clk) begin
        if (rst_n && (mon_en != 0) && l4_feat_we) begin
            nfeat = nfeat + 1;
            sum4  = sum4 + pc8(l4_feat);
            h4    <= h4 * 33 + {24'd0, l4_feat};
            if (l4_feat !== ref_feat[l4_feat_blk]) begin
                if (nbad_feat < 5)
                    $display("       L4 feat blk=%0d got %02x exp %02x", l4_feat_blk, l4_feat, ref_feat[l4_feat_blk]);
                nbad_feat = nbad_feat + 1;
            end
        end
    end

    // ---- L5 per-beat: feat8 mapping + TB-side accumulation (2048 features) ----
    always @(posedge clk) begin
        if (rst_n && (mon_en != 0) && !dir_mode && (state == S_L5) && (cyc_cnt <= 13'd255)) begin
            l5q = cyc_cnt[7:0];
            for (cc2 = 0; cc2 < 8; cc2 = cc2 + 1)
                l5e_feat[cc2] = ref_feat[((l5q % 32) * 8) + cc2][l5q / 32];
            nchk_l5b = nchk_l5b + 1;
            if (fc_feat !== l5e_feat) begin
                if (nbad_l5b < 5)
                    $display("       L5 beat %0d feat8 got %02x exp %02x", l5q, fc_feat, l5e_feat);
                nbad_l5b = nbad_l5b + 1;
            end
            for (cc2 = 0; cc2 < 4; cc2 = cc2 + 1)
                tb_acc[cc2] = tb_acc[cc2] + pc8(~(fc_feat ^ WFA[cc2][l5q*8 +: 8]));
        end
    end

    // ---- L5 result ----
    always @(posedge clk) begin
        if (rst_n && (mon_en != 0) && l5_valid && !dir_mode) begin
            nvalid5 = nvalid5 + 1;
            for (kk = 0; kk < 4; kk = kk + 1) begin
                if (l5_acc[kk*12 +: 12] !== ref_acc[kk]) begin
                    if (nbad_acc < 5)
                        $display("       L5 acc%0d got %0d exp %0d", kk, l5_acc[kk*12 +: 12], ref_acc[kk]);
                    nbad_acc = nbad_acc + 1;
                end
                if ($signed(l5_score[kk*19 +: 19]) !== ref_score[kk]) begin
                    if (nbad_sc < 5)
                        $display("       L5 score%0d got %0d exp %0d",
                                 kk, $signed(l5_score[kk*19 +: 19]), ref_score[kk]);
                    nbad_sc = nbad_sc + 1;
                end
                if (l5_acc[kk*12 +: 12] !== tb_acc[kk]) begin
                    if (nbad_acc < 5)
                        $display("       L5 acc%0d vs TB per-beat accumulator got %0d exp %0d",
                                 kk, l5_acc[kk*12 +: 12], tb_acc[kk]);
                    nbad_acc = nbad_acc + 1;
                end
            end
        end
    end

    // ---- timing / protocol ----
    always @(posedge clk) begin
        if (rst_n && (mon_en != 0)) begin
            n_st[state] = n_st[state] + 1;
            if (seq_start)              t_start = abs_cyc;
            if (all_done)               t_done  = abs_cyc;
            if (layer_done) begin
                n_done = n_done + 1;
                if (done_layer != n_done) n_bad_order = n_bad_order + 1;
                if ((done_layer >= 1) && (done_layer <= 5)) lb_seen[done_layer-1] = layer_beats;
            end
            if (state == S_L1_DONE) n_barrier = n_barrier + 1;
            if (wr_en && (wr_lo4 == 1'b0) && (wr_bank == 1'b0)) t_l1_wr_last = abs_cyc;
            if (rd_en && (state == S_L2) && (t_l2_rd_first < 0)) t_l2_rd_first = abs_cyc;
            if (din_we) begin
                if (layer == 3'd1) ndin1 = ndin1 + 1;
                if (layer == 3'd2) ndin2 = ndin2 + 1;
            end
            if (wb_req) begin
                if (layer == 3'd1) nreq1 = nreq1 + 1;
                if (layer == 3'd2) nreq2 = nreq2 + 1;
            end
            if ((layer == 3'd1) && (state == S_L1))
                if (wb_req !== din_we) nproto_bad = nproto_bad + 1;
            if ((layer == 3'd2) && (state == S_L2))
                if (wb_req !== din_we) nproto_bad = nproto_bad + 1;
            if (wb_done) nbuf_done = nbuf_done + 1;
        end
    end

    // ================================================================ load helpers
    task th_prog(input sell, input [2:0] chn, input [3:0] val);
        begin
            @(negedge clk);
            th_we   = 1'b1;
            th_sel  = sell;
            th_ch   = chn;
            th_data = val;
        end
    endtask

    task fc_w_load_uniq(input [31:0] wv);    // same 4-class byte pattern in all 256 words
        integer q;
        begin
            for (q = 0; q < 256; q = q + 1) begin
                @(negedge clk);
                wf_we   = 1'b1;
                wf_addr = q[7:0];
                wf_data = wv;
            end
            @(negedge clk);
            wf_we = 1'b0;
        end
    endtask

    task fc_w_load_py;                       // load WF0..3 (class-major blobs) as 32 bit words
        integer q, k;
        begin
            for (q = 0; q < 256; q = q + 1) begin
                @(negedge clk);
                wf_we   = 1'b1;
                wf_addr = q[7:0];
                for (k = 0; k < 4; k = k + 1)
                    wf_data[k*8 +: 8] = WFA[k][q*8 +: 8];
            end
            @(negedge clk);
            wf_we = 1'b0;
        end
    endtask

    task fc_th_load(input [15:0] t0, t1, t2, t3);
        begin
            @(negedge clk); tf_we = 1'b1; tf_ch = 2'd0; tf_data = t0;
            @(negedge clk); tf_ch = 2'd1; tf_data = t1;
            @(negedge clk); tf_ch = 2'd2; tf_data = t2;
            @(negedge clk); tf_ch = 2'd3; tf_data = t3;
            @(negedge clk); tf_we = 1'b0;
        end
    endtask

    // ================================================================ e2e frame run
    task run_frame(input integer full);
        begin
            @(negedge clk);
            mon_en = 1; dir_mode = 0;
            nbad_win1 = 0; nbad_win2 = 0; nwin1 = 0; nwin2 = 0;
            nbad_a1 = 0; nbad_a2 = 0; nbad_a3 = 0; nbad_feat = 0; nbad_sc = 0; nbad_acc = 0;
            nw1 = 0; nw2 = 0; nw3 = 0; nfeat = 0; nvalid5 = 0;
            sum1 = 0; sum2 = 0; sum3 = 0; sum4 = 0;
            ndin1 = 0; ndin2 = 0; nreq1 = 0; nreq2 = 0; nproto_bad = 0; nbuf_done = 0;
            n_done = 0; n_bad_order = 0; n_barrier = 0;
            t_start = -1; t_done = -1; t_l1_wr_last = -1; t_l2_rd_first = -1;
            h1 = 32'd0; h2 = 32'd0; h3 = 32'd0; h4 = 32'd0;
            lb_seen[0] = -1; lb_seen[1] = -1; lb_seen[2] = -1; lb_seen[3] = -1; lb_seen[4] = -1;
            for (kk = 0; kk < 8; kk = kk + 1) n_st[kk] = 0;
            // patch2 per-beat monitor state
            nbad_l3c = 0; nchk_l3c = 0; nbad_l4s = 0; nchk_l4s = 0;
            nbad_l5b = 0; nchk_l5b = 0;
            pexp_v = 0; pexp = 32'd0; l4pc = 0; l4e_v = 0; l4e_feat = 8'd0;
            for (kk = 0; kk < 8; kk = kk + 1) l4sum[kk] = 0;
            for (kk = 0; kk < 4; kk = kk + 1) tb_acc[kk] = 0;

            @(negedge clk);
            seq_start = 1'b1;
            @(negedge clk);
            seq_start = 1'b0;

            wait (all_done == 1'b1);
            @(negedge clk);
            @(negedge clk);
            mon_en = 0;
            frame_cyc = t_done - t_start;

            if (full == 0) begin
                // ---- restart pass: same image, only the key invariants ----
                $display("[tb] ---- restart frame: %0d cycles ----", frame_cyc);
                chk("restart: L1/L2 window mismatches", nbad_win1 + nbad_win2, 0);
                chk("restart: L1+L2+L3 write mismatches", nbad_a1 + nbad_a2 + nbad_a3, 0);
                chk("restart: L3 per-beat count mismatches", nbad_l3c, 0);
                chk("restart: L4 block-sum mismatches", nbad_l4s, 0);
                chk("restart: L4 feature mismatches", nbad_feat, 0);
                chk("restart: L5 per-beat feat8 mismatches", nbad_l5b, 0);
                chk("restart: L5 acc+score mismatches", nbad_acc + nbad_sc, 0);
                chk("restart: total frame cycles", frame_cyc, EXP_FRAME_CYC);
                chk("restart: L1 beat count", lb_seen[0], 4225);
                chk("restart: L3 beat count", lb_seen[2], 4096);
                chk("restart: L5 beat count", lb_seen[4], 256);
                chk("restart: layer_done order errors", n_bad_order, 0);
                chk32("restart: L1 hash", h1, EXP_H1);
                chk32("restart: L3 hash", h3, EXP_H3);
                chk32("restart: L4 hash", h4, EXP_H4);
            end else begin

            $display("[tb] ---- end-to-end frame: %0d cycles (%0d ns @100MHz) ----",
                     frame_cyc, frame_cyc * 10);

            // ---------------- windows (tests 1 and 2) ----------------
            chk("win_buf L1-mode windows (din={8{frame_bit}})", nwin1, 4096);
            chk("win_buf L1-mode window mismatches", nbad_win1, 0);
            chk("win_buf L2-mode windows (8 planes)", nwin2, 4096);
            chk("win_buf L2-mode window mismatches", nbad_win2, 0);
            // ---------------- layer outputs ----------------
            chk("ACT_A (L1) writes", nw1, 4096);
            chk("ACT_A (L1) bit mismatches", nbad_a1, 0);
            chk("ACT_B (L2) writes", nw2, 4096);
            chk("ACT_B (L2) bit mismatches", nbad_a2, 0);
            chk("ACT_A[7:0] (L3) writes", nw3, 4096);
            chk("ACT_A[7:0] (L3) bit mismatches", nbad_a3, 0);
            chk("L3 per-beat count checks (8ch x 4096)", nchk_l3c, 4096);
            chk("L3 per-beat count mismatches", nbad_l3c, 0);
            chk("L4 feature pulses", nfeat, 256);
            chk("L4 block-sum checks (8ch x 256)", nchk_l4s, 256);
            chk("L4 block-sum mismatches", nbad_l4s, 0);
            chk("L4 feature mismatches", nbad_feat, 0);
            chk("L5 per-beat feat8 checks (2048 feats)", nchk_l5b, 256);
            chk("L5 per-beat feat8 mismatches", nbad_l5b, 0);
            chk("L5 valid pulses", nvalid5, 1);
            chk("L5 acc mismatches (ref + TB accumulator)", nbad_acc, 0);
            chk("L5 score mismatches", nbad_sc, 0);
            // ---------------- Python aggregates ----------------
            chk("L1 act ones (python)", sum1, EXP_SUM1);
            chk("L2 act ones (python)", sum2, EXP_SUM2);
            chk("L3 act ones (python, 8ch)", sum3, EXP_SUM3);
            chk("L4 feat ones (python, 8ch)", sum4, EXP_SUM4);
            chk32("L1 write hash (python)", h1, EXP_H1);
            chk32("L2 write hash (python)", h2, EXP_H2);
            chk32("L3 write hash (python, 8ch)", h3, EXP_H3);
            chk32("L4 write hash (python, 8ch)", h4, EXP_H4);
            chk("L5 acc0 (python, Q6)", l5_acc[0*12 +: 12], EXP_ACC0);
            chk("L5 acc1 (python, Q6)", l5_acc[1*12 +: 12], EXP_ACC1);
            chk("L5 acc2 (python, Q6)", l5_acc[2*12 +: 12], EXP_ACC2);
            chk("L5 acc3 (python, Q6)", l5_acc[3*12 +: 12], EXP_ACC3);
            chk("L5 score0 (python, Q6 19b)", $signed(l5_score[0*19 +: 19]), EXP_SC0);
            chk("L5 score1 (python, Q6 19b)", $signed(l5_score[1*19 +: 19]), EXP_SC1);
            chk("L5 score2 (python, Q6 19b)", $signed(l5_score[2*19 +: 19]), EXP_SC2);
            chk("L5 score3 (python, Q6 19b)", $signed(l5_score[3*19 +: 19]), EXP_SC3);
            // ---------------- timing (patch2 鍙ｅ緞) ----------------
            chk("layer beats L1 (65x65)", lb_seen[0], 4225);
            chk("layer beats L2 (65x65)", lb_seen[1], 4225);
            chk("layer beats L3 (1x1, 8ch)", lb_seen[2], 4096);
            chk("layer beats L4 (4x4 pool, 8ch)", lb_seen[3], 4096);
            chk("layer beats L5 (FC 2048)", lb_seen[4], 256);
            chk("layer_done pulses", n_done, 5);
            chk("layer_done order errors", n_bad_order, 0);
            chk("S_L1 state cycles", n_st[S_L1], EXP_L1_CYC);
            chk("S_L1_DONE barrier cycles", n_st[S_L1_DONE], 1);
            chk("S_L2 state cycles", n_st[S_L2], EXP_L2_CYC);
            chk("S_L3 state cycles", n_st[S_L3], EXP_L3_CYC);
            chk("S_L4 state cycles", n_st[S_L4], EXP_L4_CYC);
            chk("S_L5 state cycles (256+1)", n_st[S_L5], EXP_L5_CYC);
            chk("S_DONE state cycles", n_st[S_DONE], 1);
            chk("sum of state cycles", n_st[S_L1] + n_st[S_L1_DONE] + n_st[S_L2] +
                n_st[S_L3] + n_st[S_L4] + n_st[S_L5] + n_st[S_DONE], EXP_FRAME_CYC);
            chk("total frame cycles", frame_cyc, EXP_FRAME_CYC);
            chk("L1 warm-up beats (beats - windows)", lb_seen[0] - nwin1, 129);
            chk("L2 warm-up beats (beats - windows)", lb_seen[1] - nwin2, 129);
            chk("L1 data pixel requests", ndin1, 4096);
            chk("L2 data pixel requests", ndin2, 4096);
            chk("L1 win_buf din_req pulses", nreq1, 4096);
            chk("L2 win_buf din_req pulses", nreq2, 4096);
            chk("din_req != din_we cycles", nproto_bad, 0);
            chk("win_buf done pulses", nbuf_done, 2);
            if ((t_l2_rd_first > t_l1_wr_last) && (t_l1_wr_last > 0) && (n_barrier == 1))
                chk("L1->L2 barrier: first L2 read after last L1 write", 1, 1);
            else
                chk("L1->L2 barrier: first L2 read after last L1 write", 0, 1);
            $display("[tb] timing: t_start=%0d t_done=%0d last_L1_wr=%0d first_L2_rd=%0d barrier=%0d",
                     t_start, t_done, t_l1_wr_last, t_l2_rd_first, n_barrier);
            $display("[tb] frame = %0d cycles = %0d ns @50MHz -> %0d fps",
                     frame_cyc, frame_cyc * 20,
                     (frame_cyc == 0) ? 0 : ((1000000000 + frame_cyc*10) / (frame_cyc * 20)));
            $display("[tb] patch2 口径: L1 4228 + barrier 1 + L2 4228 + L3 4098 + L4 4098 + L5 257 + done 1 = 16911");
            end   // if (full)
        end
    endtask

    // ================================================================ test 4: l3_mix
    task l3_probe(input [7:0] din, input [63:0] w3v, input [31:0] th3v);
        begin
            @(negedge clk);
            dir_mode  = 1'b1;
            tb_l3_din = din;
            tb_w3     = w3v;
            tb_th3    = th3v;
            tb_l3_en  = 1'b1;
            @(negedge clk);
            tb_l3_en  = 1'b0;
        end
    endtask

    task l3_expect(input [7:0] din, input [63:0] w3v, input [31:0] th3v,
                   output [31:0] e_cnt, output [7:0] e_act);
        integer k;
        reg [3:0] pc;
        begin
            for (k = 0; k < 8; k = k + 1) begin
                pc = pc8(~(din ^ w3v[k*8 +: 8]));
                e_cnt[k*4 +: 4] = pc;
                e_act[k] = (pc > th3v[k*4 +: 4]);
            end
        end
    endtask

    task test_l3_mix;
        integer i, nbad_l3;
        reg [31:0] e_cnt;
        reg [7:0]  e_act;
        reg [31:0] lfsr;
        reg [7:0]  din;
        reg [63:0] rw3;
        reg [31:0] rth3;
        begin
            $display("[tb] ---- test 4: l3_mix (8 channels) ----");
            // ---- python vector 0: din=FF w3=0 -> cnt 0 x8 ----
            l3_probe(8'hFF, 64'h0, TH3V);
            chk32("l3 cnt python v0", l3_cnt, 32'h0);
            chk("l3 act python v0", l3_act, 8'h00);
            // ---- python vector 1: din=FF w3=all ones -> cnt 8 x8 ----
            l3_probe(8'hFF, 64'hFFFF_FFFF_FFFF_FFFF, TH3V);
            chk32("l3 cnt python v1", l3_cnt, 32'h8888_8888);
            chk("l3 act python v1", l3_act, 8'hFF);
            // ---- python vector 2: din=5A w3=3C69A5F0DEADBEEF -> cnt 3,4,1,6,4,0,4,4 ----
            l3_probe(8'h5A, L3V2_W, TH3V);
            chk32("l3 cnt python v2", l3_cnt, 32'h4404_6143);
            chk("l3 act python v2", l3_act, L3V2_A);
            // ---- python vector 3: din=00 w3=A55A3CC3F00F0FF0, th3=[4,3,5,2,8,0,7,1] ----
            l3_probe(8'h00, L3V3_W, 32'h1708_2534);
            chk32("l3 cnt python v3", l3_cnt, L3V3_C);
            chk("l3 act python v3", l3_act, L3V3_A);
            // ---- 16 random vectors vs an independent popcount8 on all 8 channels ----
            lfsr = 32'hC0DE_1234;
            nbad_l3 = 0;
            for (i = 0; i < 16; i = i + 1) begin
                lfsr = (lfsr << 1) ^ (lfsr[31] ? 32'h04C1_1DB7 : 32'd0);
                din  = lfsr[7:0] ^ lfsr[23:16];
                rw3  = {lfsr, ~lfsr ^ {lfsr[15:0], lfsr[31:16]}};
                rth3 = {lfsr[3:0], lfsr[11:8], lfsr[19:16], lfsr[27:24],
                        lfsr[7:4], lfsr[15:12], lfsr[23:20], lfsr[31:28]} & 32'h8888_8888;
                l3_probe(din, rw3, rth3);
                l3_expect(din, rw3, rth3, e_cnt, e_act);
                if (l3_cnt !== e_cnt) begin
                    $display("       l3 random %0d cnt got %08x exp %08x", i, l3_cnt, e_cnt);
                    nbad_l3 = nbad_l3 + 1;
                end else if (l3_act !== e_act) begin
                    $display("       l3 random %0d act got %02x exp %02x", i, l3_act, e_act);
                    nbad_l3 = nbad_l3 + 1;
                end
            end
            if (nbad_l3 == 0) begin
                $display("[PASS] l3_mix 16 random vectors x 8 channels (cnt+act) vs popcount8");
                checks = checks + 1;
            end else begin
                $display("[FAIL] l3_mix random vectors : %0d/16 bad", nbad_l3);
                errors = errors + 1;
            end
            // restore the e2e vectors
            tb_w3  = W3;
            tb_th3 = TH3V;
        end
    endtask

    // ================================================================ test 5: l4_pool
    reg [7:0] l4_mon_feat;
    reg [7:0] l4_mon_blk;
    integer   l4_mon_cnt;
    reg [7:0] l4_hist [0:7];

    always @(posedge clk) begin
        if (rst_n && dir_mode && l4_feat_we) begin
            l4_mon_feat = l4_feat;
            l4_mon_blk  = l4_feat_blk;
            l4_mon_cnt  = l4_mon_cnt + 1;
        end
    end

    task l4_sof_pulse;
        begin
            @(negedge clk); tb_l4_sof = 1'b1;
            @(negedge clk); tb_l4_sof = 1'b0;
        end
    endtask

    task l4_block(input [127:0] pat);
        integer j, c;
        begin
            for (j = 0; j < 16; j = j + 1) begin
                @(negedge clk);
                tb_l4_en = 1'b1;
                for (c = 0; c < 8; c = c + 1) tb_l4_din[c] = pat[c*16 + j];
            end
            @(negedge clk);
            tb_l4_en = 1'b0;
            @(negedge clk);
        end
    endtask

    task l4_check_block(input [8*40-1:0] name, input [127:0] pat, input [39:0] th);
        reg [7:0] e;
        begin
            e = l4_ref(pat, th);
            if (l4_mon_feat !== e) begin
                $display("[FAIL] %0s : got %02x exp %02x (blk %0d)", name, l4_mon_feat, e, l4_mon_blk);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s : feat %02x (blk %0d)", name, l4_mon_feat, l4_mon_blk);
                checks = checks + 1;
            end
        end
    endtask

    task test_l4_pool;
        integer ch, j;
        reg [7:0] e;
        begin
            $display("[tb] ---- test 5: l4_pool (8 channels, 4x4 blocks) ----");
            dir_mode = 1'b1;
            l4_mon_cnt = 0;
            tb_th4 = TH4V;                     // 8 x 5 bit, TH4 = 8 -> >=8/16
            l4_sof_pulse;

            // blocks 0..7 -> packed word 0 (8 blocks x 8 channels = 64 bit)
            l4_block({16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF});
            l4_hist[0] = l4_mon_feat;
            l4_check_block("l4 all-ones 16/16", {16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF,16'hFFFF}, TH4V);
            l4_block(128'h0);
            l4_hist[1] = l4_mon_feat;
            l4_check_block("l4 all-zero 0/16", 128'h0, TH4V);
            l4_block({16'h807F,16'hFFFF,16'h00FF,16'hFF00,16'hAAAA,16'h5555,16'h0001,16'h8000});
            l4_hist[2] = l4_mon_feat;
            l4_check_block("l4 mixed 7/16/8/0", {16'h807F,16'hFFFF,16'h00FF,16'hFF00,16'hAAAA,16'h5555,16'h0001,16'h8000}, TH4V);
            l4_block({16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF});
            l4_hist[3] = l4_mon_feat;
            l4_check_block("l4 15-of-16 -> 1", {16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF,16'h7FFF}, TH4V);
            l4_block({16'h00FF,16'h00FF,16'h00FF,16'h00FF,16'h00FF,16'h00FF,16'h00FF,16'h00FF});
            l4_hist[4] = l4_mon_feat;
            l4_check_block("l4 8-of-16 -> 1", {16'h00FF,16'h00FF,16'h00FF,16'h00FF,16'h00FF,16'h00FF,16'h00FF,16'h00FF}, TH4V);
            l4_block({16'h007F,16'h007F,16'h007F,16'h007F,16'h007F,16'h007F,16'h007F,16'h007F});
            l4_hist[5] = l4_mon_feat;
            l4_check_block("l4 7-of-16 -> 0", {16'h007F,16'h007F,16'h007F,16'h007F,16'h007F,16'h007F,16'h007F,16'h007F}, TH4V);
            l4_block({16'h0001,16'h0002,16'h0004,16'h0008,16'h0010,16'h0020,16'h0040,16'h0080});
            l4_hist[6] = l4_mon_feat;
            l4_check_block("l4 single-bit", {16'h0001,16'h0002,16'h0004,16'h0008,16'h0010,16'h0020,16'h0040,16'h0080}, TH4V);
            l4_block({16'hFFFF,16'hAAAA,16'h5555,16'h0000,16'hFFFF,16'hAAAA,16'h5555,16'h0000});
            l4_hist[7] = l4_mon_feat;
            l4_check_block("l4 alternating ch", {16'hFFFF,16'hAAAA,16'h5555,16'h0000,16'hFFFF,16'hAAAA,16'h5555,16'h0000}, TH4V);

            chk("l4 feat pulses after 8 blocks", l4_mon_cnt, 8);
            chk("l4 last block index", l4_mon_blk, 7);
            // ---- packed word 0 readback through the L5 read port (all 8 channels) ----
            tb_frd_addr = 5'd0;
            for (ch = 0; ch < 8; ch = ch + 1) begin
                tb_frd_ch = ch[2:0];
                #1;
                e = 8'd0;
                for (j = 0; j < 8; j = j + 1) e[j] = l4_hist[j][ch];
                if (l4_frd_data !== e) begin
                    $display("[FAIL] l4 packed word0 ch%0d : got %02x exp %02x", ch, l4_frd_data, e);
                    errors = errors + 1;
                end else begin
                    $display("[PASS] l4 packed word0 ch%0d : %02x", ch, l4_frd_data);
                    checks = checks + 1;
                end
            end
            // ---- non-default threshold: 12/16 ----
            tb_th4 = {5'd12,5'd12,5'd12,5'd12,5'd12,5'd12,5'd12,5'd12};
            l4_sof_pulse;
            l4_mon_cnt = 0;
            l4_block({16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0FFF});
            l4_check_block("l4 th=12 12/16 -> 1", {16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0FFF}, {5'd12,5'd12,5'd12,5'd12,5'd12,5'd12,5'd12,5'd12});
            l4_block({16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h07FF});
            l4_check_block("l4 th=12 11/16 -> 0", {16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h0000,16'h07FF}, {5'd12,5'd12,5'd12,5'd12,5'd12,5'd12,5'd12,5'd12});
            chk("l4 feat pulses th test", l4_mon_cnt, 2);
            tb_th4 = TH4V;
            dir_mode = 1'b0;
        end
    endtask

    // ================================================================ test 6: l5_fc
    task l5_run(input [7:0] feat);
        begin
            @(negedge clk);
            dir_mode    = 1'b1;
            tb_fc_feat  = feat;
            tb_fc_start = 1'b1;
            @(negedge clk);
            tb_fc_start = 1'b0;
            wait (l5_valid == 1'b1);
            @(negedge clk);
        end
    endtask

    task l5_check(input [8*44-1:0] name,
                  input integer a0, a1, a2, a3,
                  input integer e0, e1, e2, e3);
        integer ok;
        begin
            ok = 1;
            if (l5_acc[0*12 +: 12] !== a0) ok = 0;
            if (l5_acc[1*12 +: 12] !== a1) ok = 0;
            if (l5_acc[2*12 +: 12] !== a2) ok = 0;
            if (l5_acc[3*12 +: 12] !== a3) ok = 0;
            if ($signed(l5_score[0*19 +: 19]) !== e0) ok = 0;
            if ($signed(l5_score[1*19 +: 19]) !== e1) ok = 0;
            if ($signed(l5_score[2*19 +: 19]) !== e2) ok = 0;
            if ($signed(l5_score[3*19 +: 19]) !== e3) ok = 0;
            if (!ok) begin
                $display("[FAIL] %0s : acc %0d %0d %0d %0d score %0d %0d %0d %0d",
                         name, l5_acc[0*12 +: 12], l5_acc[1*12 +: 12],
                         l5_acc[2*12 +: 12], l5_acc[3*12 +: 12],
                         $signed(l5_score[0*19 +: 19]), $signed(l5_score[1*19 +: 19]),
                         $signed(l5_score[2*19 +: 19]), $signed(l5_score[3*19 +: 19]));
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s : acc %0d %0d %0d %0d score %0d %0d %0d %0d",
                         name, a0, a1, a2, a3, e0, e1, e2, e3);
                checks = checks + 1;
            end
        end
    endtask

    task test_l5_fc;
        integer s0, s1, s2, s3, best, bi, s9a, s9b;
        begin
            $display("[tb] ---- test 6: l5_fc (2048 -> 4, Q6) ----");
            dir_mode = 1'b1;
            // ---- A: python "l5 case A" -- weight byte 0x5A, feat 0xFF -> 4 matches/beat
            //         acc = 256*4 = 1024, score = 64*(2048-2048) + bias = bias
            fc_w_load_uniq(32'h5A5A_5A5A);
            fc_th_load(16'd61, 16'd62, 16'hFFFD, 16'h8000);       // 61, 62, -3, -32768
            l5_run(8'hFF);
            l5_check("l5 A acc1024 Q6 bias", 1024, 1024, 1024, 1024, 61, 62, -3, -32768);
            // ---- B: perfect match (feat == weight) -> acc 2048, bias +32767 -> 163839
            fc_w_load_uniq(32'h5A5A_5A5A);
            fc_th_load(16'h7FFF, 16'h7FFF, 16'h7FFF, 16'h7FFF);
            l5_run(8'h5A);
            l5_check("l5 B acc2048 max score", 2048, 2048, 2048, 2048, 163839, 163839, 163839, 163839);
            // ---- C: no match (feat = ~weight) -> acc 0, bias -32768 -> -163840
            fc_w_load_uniq(32'hA5A5_A5A5);
            fc_th_load(16'h8000, 16'h8000, 16'h8000, 16'h8000);
            l5_run(8'h5A);
            l5_check("l5 C acc0 min score", 0, 0, 0, 0, -163840, -163840, -163840, -163840);
            // ---- E: 16 bit sign extension around zero (bias -1/0/+1/-2) ----
            fc_w_load_uniq(32'hA5A5_A5A5);
            fc_th_load(16'hFFFF, 16'h0000, 16'h0001, 16'hFFFE);
            l5_run(8'h5A);
            l5_check("l5 E bias sign extend", 0, 0, 0, 0, -131073, -131072, -131071, -131074);
            // ---- D: E6 near tie -- class0/1 match counts are EQUAL (acc 2048),
            //         float bias 0.95 vs 0.97 -> bias_q 61 vs 62 -> score differs by 1 LSB
            fc_w_load_uniq(32'hA5A5_5A5A);                       // ch0/1 = 5A (match), ch2/3 = A5
            fc_th_load(16'd61, 16'd62, 16'd0, 16'd0);
            l5_run(8'h5A);
            l5_check("l5 D near tie 0.95/0.97", 2048, 2048, 0, 0, 131133, 131134, -131072, -131072);
            s0 = $signed(l5_score[0*19 +: 19]);
            s1 = $signed(l5_score[1*19 +: 19]);
            chk("Q6 near-tie: score1 - score0", s1 - s0, 1);
            // argmax must be class 1 (float bias 0.97 > 0.95)
            best = 0;
            for (bi = 1; bi < 4; bi = bi + 1)
                if ($signed(l5_score[bi*19 +: 19]) > $signed(l5_score[best*19 +: 19])) best = bi;
            chk("Q6 near-tie: argmax = class1", best, 1);
            // what the OLD 9 bit integer bias would have decided on this pair:
            //   bias_int = round(bias_q / 64) = round(61/64) = 1 and round(62/64) = 1
            //   -> score_int(0) == score_int(1) -> exact tie, argmax falls back to class 0
            s9a = (2*2048 - 2048) + ((61 + 32) / 64);
            s9b = (2*2048 - 2048) + ((62 + 32) / 64);
            chk("int9 emulation ties on this pair (Q6 is needed)", s9b - s9a, 0);
            $display("[tb] Q6 demo: acc0=acc1=2048, bias_q 61 vs 62 -> score %0d vs %0d (delta %0d);",
                     s0, s1, s1 - s0);
            $display("[tb]           old int9 bias round(0.95)=round(0.97)=1 -> tie, argmax would be 0");
            // ---- restore the python vectors ----
            fc_w_load_py;
            fc_th_load(THFC[0], THFC[1], THFC[2], THFC[3]);
            dir_mode = 1'b0;
        end
    endtask

    // ================================================================ main
    integer i;
    initial begin
        errors = 0;
        checks = 0;

        // defaults
        rst_n = 1'b0; seq_start = 1'b0; dir_mode = 0;
        th_we = 0; th_sel = 0; th_ch = 0; th_data = 0;
        wf_we = 0; wf_addr = 0; wf_data = 0;
        tf_we = 0; tf_ch = 0; tf_data = 0;
        tb_l3_en = 0; tb_l3_din = 0; tb_w3 = W3; tb_th3 = 32'd0;
        tb_l4_en = 0; tb_l4_din = 0; tb_l4_sof = 0; tb_th4 = 40'd0;
        tb_fc_start = 0; tb_fc_feat = 0;
        tb_frd_addr = 0; tb_frd_ch = 0;
        mon_en = 0; abs_cyc = 0; t_start = -1; t_done = -1;
        t_l1_wr_last = -1; t_l2_rd_first = -1; frame_cyc = 0;
        l4_mon_cnt = 0; l4_mon_feat = 0; l4_mon_blk = 0;
        h1 = 0; h2 = 0; h3 = 0; h4 = 0;
        for (i = 0; i < 8; i = i + 1) n_st[i] = 0;
        for (i = 0; i < 8; i = i + 1) l4_hist[i] = 0;

        TH1[0] = 4'd4; TH1[1] = 4'd4; TH1[2] = 4'd5; TH1[3] = 4'd3;
        TH1[4] = 4'd6; TH1[5] = 4'd2; TH1[6] = 4'd4; TH1[7] = 4'd5;
        TH2[0] = 4'd5; TH2[1] = 4'd4; TH2[2] = 4'd4; TH2[3] = 4'd6;
        TH2[4] = 4'd3; TH2[5] = 4'd5; TH2[6] = 4'd4; TH2[7] = 4'd4;
        TH3[0] = 4'd4; TH3[1] = 4'd3; TH3[2] = 4'd5; TH3[3] = 4'd4;
        TH3[4] = 4'd6; TH3[5] = 4'd2; TH3[6] = 4'd5; TH3[7] = 4'd4;
        TH4[0] = 5'd8; TH4[1] = 5'd8; TH4[2] = 5'd8; TH4[3] = 5'd8;
        TH4[4] = 5'd8; TH4[5] = 5'd8; TH4[6] = 5'd8; TH4[7] = 5'd8;
        // Q6: bias_float = [-0.0252, -0.0497, -0.0294, 0.0149] -> round(x*64)
        THFC[0] = -16'sd2; THFC[1] = -16'sd3; THFC[2] = -16'sd2; THFC[3] = 16'sd1;
        build_thv;
        tb_th3 = TH3V;
        tb_th4 = TH4V;

        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        repeat (3) @(negedge clk);

        $display("[tb] ===== patch2 datapath: layer_seq / l3_mix 8ch / l4_pool 8ch / l5_fc 2048 Q6 =====");

        // ---- stimulus + reference (before the DUT run) ----
        build_frame;
        build_ref;
        print_ref_summary;

        // ---- program L1/L2 theta (8 channels each, 4 bit unsigned) ----
        for (i = 0; i < 8; i = i + 1) th_prog(1'b0, i[2:0], TH1[i]);
        for (i = 0; i < 8; i = i + 1) th_prog(1'b1, i[2:0], TH2[i]);
        @(negedge clk); th_we = 1'b0; th_sel = 1'b0;

        // ---- program L3 / L4 theta + L5 weights (Q6 bias) ----
        fc_w_load_py;
        fc_th_load(THFC[0], THFC[1], THFC[2], THFC[3]);

        // ---- end-to-end frame ----
        run_frame(1);

        // ---- restart from S_IDLE (bank reuse / re-initialisation / q_eff) ----
        run_frame(0);

        // ---- directed module tests ----
        test_l3_mix;
        test_l4_pool;
        test_l5_fc;

        // ---- summary ----
        $display("================================================================");
        if (errors == 0) $display("[RESULT] ALL PASS (%0d checks)", checks);
        else             $display("[RESULT] %0d FAILURE(S) (%0d checks)", errors, checks);
        $display("================================================================");
        $finish;
    end

    // global timeout
    initial begin
        #5000000;
        $display("[FAIL] global timeout (5 ms)");
        $finish;
    end

endmodule
