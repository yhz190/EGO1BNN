// bnn_engine.v -- D4/G1 engine datapath top: layer_seq + win_buf + lane_array + accum
//                 + 2 x theta_cmp + act_bram + l3_mix + l4_pool + l5_fc
//
// This is the *integrating* wrapper around the already signed-off modules.  None of
// popcnt9 / win_buf / lane_array / accum / theta_cmp / act_bram / l3_mix / l4_pool /
// l5_fc / layer_seq is modified; the wiring below is the one verified by
// tb_pipeline.v (patch2, 16,911 cycles / 2,957 fps), only the test-mode muxes of the
// TB are removed and the real weight/theta/frame ports are added.
//
// ============================== frame / ACT ping-pong ==============================
//   L1  FRAME (2 banks x 4096 x 1 bit, prefix-read)  -> ACT_A
//   L2  ACT_A -> ACT_B
//   L3  ACT_B -> ACT_A[7:0]
//   L4  ACT_A[7:0] -> l4_pool internal 2048 bit feature RAM
//   L5  2048 -> 4 scores (Q6, 19 bit signed)
//
//   FRAME is double buffered (D5): the capture side always writes the *inactive*
//   bank (frame_bank = ~active), while L1 reads the bank that was active when `start`
//   was accepted (frame_rd_bank, latched into frm_rd_bank_r).  A new 517 B frame may
//   therefore be streamed in while a frame is being computed without corrupting it.
//   FRAME and ACT_A still share bank A, so FRAME writes are only meaningful while
//   the engine is idle or while it runs on the other bank (which is always the case).
//
// ============================== read prefetch =====================================
//   act_bram has a registered read (1 cycle).  layer_seq already presents the address
//   of the pixel that is consumed on the NEXT cycle, so ACT data can be used directly.
//   FRAME is registered too, so its address must be pix_idx + din_we (i.e. pixcnt of
//   the next cycle, exactly the rd_nx13 expression inside layer_seq).  Both therefore
//   reproduce the timing of tb_pipeline (where FRAME was an async array read).
//
// ============================== weight / theta load ==============================
//   w_we/w_addr/w_data is the 271-word (8,648 bit) blob of
//   src/weights_layout_spec.md v5, bit i -> word i/32, bit i%32 (LSB first
//   within the word, the authoritative order of scripts/gen_weights.py).  The words
//   MUST arrive in 271 CONSECUTIVE cycles.
//
//   v5 stores the FC field *beat-major*, i.e. exactly the 32-bit words l5_fc wants on
//   beat q = {b3,b2,b1,b0}.  bnn_wblob therefore streams those words straight into
//   l5_fc's 256x32 wmem as they arrive (one write at blob word 7+q, the cycle the beat
//   word's low half shows up); there is no fc_ram cache and no de-interleave pass any
//   more (that was 2,769 LUT / 8,788 FF after synthesis -- see
//   sim/20261101_beatmajor/).  Load cost is 271 blob words + a 36 cycle theta push.
//   load_done ( = ready) rises at the end; the CPU must not issue bnn.conv before that
//   (top_bnn holds the CPU in reset until then).
//
//   th_we/th_sel/th_ch/th_data is a direct theta write path (debug / directed tests),
//   same encoding as the push: sel 0=theta1 1=theta2 2=theta3 3=theta4 4=thetaFC.
//
// Verilog-2001, synthesizable, no DSP.
`timescale 1ns / 1ps

module bnn_engine (
    input  wire        clk,
    input  wire        rst_n,        // async active-low
    input  wire        start,        // 1-cycle pulse: run one full frame (L1..L5)
    output wire        busy,         // layer_seq busy (S_IDLE..S_DONE)
    output wire        all_done,     // 1-cycle pulse in S_DONE, score_o already stable
    output wire        ready,        // weights/theta fully loaded
    // ---- FRAME write port (D5: two banks, the capture side writes the inactive one) ----
    input  wire        frame_we,
    input  wire [11:0] frame_addr,   // pixel index row*64+col (same index in both banks)
    input  wire        frame_bit,
    input  wire        frame_bank,   // bank written by frame_we (top: ~active bank)
    input  wire        frame_rd_bank,// active bank at `start` (latched for the frame)
    // ---- weight blob: 271 x 32 bit, 271 consecutive cycles ----
    input  wire        w_we,
    input  wire [8:0]  w_addr,
    input  wire [31:0] w_data,
    // ---- direct theta load ----
    input  wire        th_we,
    input  wire [2:0]  th_sel,
    input  wire [2:0]  th_ch,
    input  wire [15:0] th_data,
    // ---- results ----
    output wire [2:0]  cls_o,        // hardware argmax of score_o (ties -> lower index)
    output wire [75:0] score_o,      // class k in score_o[k*19 +: 19], signed Q6
    output wire [47:0] acc_o,        // class k in acc_o[k*12 +: 12], 0..2048
    // ---- debug ----
    output wire [2:0]  layer_o,
    output wire [2:0]  state_o,
    output wire [12:0] beat_o,       // beat count of the last finished layer
    output wire        wload_busy
);

    // ================================================================ globals
    wire [2:0]  layer, done_layer, state;
    wire        layer_start, layer_done, sof, din_we, pix_en;
    wire        rd_en, rd_bank, wr_en, wr_bank, wr_lo4;
    wire [11:0] pix_idx, rd_addr, waddr;
    wire [12:0] beat_cnt, win_cnt, cyc_cnt, layer_beats;

    assign layer_o = layer;
    assign state_o = state;
    assign beat_o  = layer_beats;
    assign busy     = (state != 3'd0);

    // declared up here because layer_seq is instantiated above the lane pipeline
    wire wb_vld;

    // ================================================================ layer sequencer
    layer_seq u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .win_valid(wb_vld),
        .busy(), .layer(layer), .layer_done(layer_done), .done_layer(done_layer),
        .all_done(all_done), .state(state), .layer_start(layer_start),
        .sof(sof), .din_we(din_we), .pix_idx(pix_idx),
        .rd_en(rd_en), .rd_bank(rd_bank), .rd_addr(rd_addr),
        .wr_en(wr_en), .wr_bank(wr_bank), .wr_lo4(wr_lo4), .waddr(waddr),
        .pix_en(pix_en), .beat_cnt(beat_cnt), .win_cnt(win_cnt),
        .cyc_cnt(cyc_cnt), .layer_beats(layer_beats)
    );

    // ================================================================ FRAME store (2 x 4096 x 1)
    // Two 512 B banks packed into one 8 Kbit array (still a single RAMB18, D5).  The
    // write port and the read port present different addresses, so the inferred
    // simple-dual-port block RAM can capture a new frame while L1 reads the old one.
    (* ram_style = "block" *) reg frm_mem [0:8191];

    wire [12:0] frm_waddr = {frame_bank, frame_addr};

    always @(posedge clk) begin
        if (frame_we) frm_mem[frm_waddr] <= frame_bit;
    end

    // read bank frozen for the whole frame (`start` is a 1-cycle pulse)
    reg frm_rd_bank_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)     frm_rd_bank_r <= 1'b0;
        else if (start) frm_rd_bank_r <= frame_rd_bank;
    end

    // registered read with the layer_seq prefetch (address of the NEXT pixel)
    wire [12:0] frm_nx13 = {1'b0, pix_idx} + {12'd0, din_we};
    wire [11:0] frm_nx12 = (frm_nx13 > 13'd4095) ? 12'd4095 : frm_nx13[11:0];
    wire [12:0] frm_raddr = {frm_rd_bank_r, frm_nx12};
    reg         frm_bit;
    always @(posedge clk) begin
        frm_bit <= frm_mem[frm_raddr];
    end

    // ================================================================ weights / theta
    wire [71:0] l1_w, l2_w;
    wire [63:0] w3;
    wire [31:0] th1_v, th2_v, th3_v;
    wire [55:0] th4_v;
    wire [63:0] thfc_v;
    wire        push_we;
    wire [2:0]  push_sel, push_ch;
    wire [15:0] push_data;
    wire        l5_w_we;
    wire [7:0]  l5_w_addr;
    wire [31:0] l5_w_data;
    wire        load_done;

    bnn_wblob u_blob (
        .clk(clk), .rst_n(rst_n),
        .w_we(w_we), .w_addr(w_addr), .w_data(w_data),
        .l1_w(l1_w), .l2_w(l2_w), .w3(w3),
        .th1_v(th1_v), .th2_v(th2_v), .th3_v(th3_v), .th4_v(th4_v), .thfc_v(thfc_v),
        .push_we(push_we), .push_sel(push_sel), .push_ch(push_ch), .push_data(push_data),
        .l5_we(l5_w_we), .l5_addr(l5_w_addr), .l5_data(l5_w_data),
        .load_done(load_done)
    );

    reg ready_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)     ready_r <= 1'b0;
        else if (load_done) ready_r <= 1'b1;
    end
    assign ready       = ready_r;
    assign wload_busy  = ~ready_r;

    // ---- theta write mux: blob push has priority over the direct port ----
    wire       thwr = push_we | th_we;
    wire [2:0] thsel = push_we ? push_sel : th_sel;
    wire [2:0] thch  = push_we ? push_ch  : th_ch;
    wire [15:0] thd  = push_we ? push_data : th_data;

    reg [31:0] th3_r;
    reg [39:0] th4_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            th3_r <= 32'd0;
            th4_r <= 40'd0;
        end else if (thwr) begin
            if (thsel == 3'd2) th3_r[thch*4 +: 4] <= thd[3:0];
            if (thsel == 3'd3) th4_r[thch*5 +: 5] <= thd[4:0];
        end
    end

    // ================================================================ window / lanes
    wire [71:0] wb_win;
    wire [31:0] lane_cnt;
    wire [95:0] acc;
    wire [7:0]  act1, act2, act8;
    wire [7:0]  act_rdata;

    wire [7:0]  wb_din   = (layer == 3'd1) ? {8{frm_bit}} : act_rdata;
    wire [71:0] lane_wgt = (layer == 3'd1) ? l1_w : l2_w;

    win_buf #(.W(8)) u_wb (
        .clk(clk), .rst_n(rst_n), .sof(sof), .din_we(din_we), .din(wb_din),
        .win(wb_win), .win_valid(wb_vld), .win_row(), .win_col(),
        .din_req(), .done()
    );

    lane_array u_la (.win(wb_win), .wgt(lane_wgt), .cnt(lane_cnt));

    accum #(.W(12)) u_acc (
        .clk(clk), .rst_n(rst_n), .clr(wb_vld), .ld(1'b0), .ld_ch(3'd0),
        .ld_data(12'd0), .en(wb_vld), .din(lane_cnt), .acc(acc)
    );

    theta_cmp #(.W_ACC(12), .W_TH(4)) u_th1 (
        .clk(clk), .rst_n(rst_n),
        .th_we(thwr && (thsel == 3'd0)), .th_ch(thch), .th_data(thd[3:0]),
        .acc(acc), .act(act1), .th_val()
    );

    theta_cmp #(.W_ACC(12), .W_TH(4)) u_th2 (
        .clk(clk), .rst_n(rst_n),
        .th_we(thwr && (thsel == 3'd1)), .th_ch(thch), .th_data(thd[3:0]),
        .acc(acc), .act(act2), .th_val()
    );

    assign act8 = (layer == 3'd1) ? act1 : act2;

    // ================================================================ ACT ping-pong
    wire [7:0]  l3_act;
    wire [31:0] l3_cnt;
    wire        l3_valid;

    wire [7:0]  act_wdata = wr_lo4 ? l3_act : act8;

    act_bram #(.P(8), .AW(12), .DEPTH(4096)) u_act (
        .clk(clk), .we(wr_en), .wbank(wr_bank), .waddr(waddr), .wdata(act_wdata),
        .re(rd_en), .rbank(rd_bank), .raddr(rd_addr), .rdata(act_rdata)
    );

    // ================================================================ L3 / L4 / L5
    wire [7:0]  l4_feat, l4_frd_data;
    wire        l4_feat_we;
    wire [7:0]  l4_feat_blk;
    wire        l5_valid;
    wire [75:0] l5_score;
    wire [47:0] l5_acc;
    wire [4:0]  l5_fa;
    wire [2:0]  l5_fch;

    l3_mix #(.COUT(8), .W_TH(4)) u_l3 (
        .clk(clk), .rst_n(rst_n),
        .en(pix_en && (layer == 3'd3)), .din(act_rdata), .w3(w3), .th3(th3_r),
        .act(l3_act), .cnt(l3_cnt), .valid(l3_valid)
    );

    l4_pool #(.NCH(8), .NF(256), .W_TH(5)) u_l4 (
        .clk(clk), .rst_n(rst_n),
        .sof(layer_start && (layer == 3'd4)),
        .en(pix_en && (layer == 3'd4)), .din(act_rdata), .th4(th4_r),
        .feat_we(l4_feat_we), .feat_blk(l4_feat_blk), .feat(l4_feat),
        .frd_addr(l5_fa), .frd_ch(l5_fch), .frd_data(l4_frd_data)
    );

    l5_fc #(.NF(2048), .NC(4), .NCH(8), .W_TH(16), .SW(19)) u_l5 (
        .clk(clk), .rst_n(rst_n),
        .start(layer_start && (layer == 3'd5)),
        .feat8(l4_frd_data), .feat_addr(l5_fa), .feat_ch(l5_fch),
        .w_we(l5_w_we), .w_addr(l5_w_addr), .w_data(l5_w_data),
        .th_we(thwr && (thsel == 3'd4)), .th_ch(thch[1:0]), .th_data(thd),
        .valid(l5_valid), .score(l5_score), .acc_out(l5_acc)
    );

    // ================================================================ result registers
    // l5_fc.valid pulses 1 cycle before layer_seq enters S_DONE (all_done), so the
    // scores are stable by the time all_done is seen.
    reg        [75:0] score_r;
    reg        [47:0] acc_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            score_r <= 76'd0;
            acc_r   <= 48'd0;
        end else if (l5_valid) begin
            score_r <= l5_score;
            acc_r   <= l5_acc;
        end
    end

    assign score_o = score_r;
    assign acc_o   = acc_r;

    // hardware argmax, signed compare, ties resolved to the lower class index
    wire signed [18:0] s0 = score_r[0*19 +: 19];
    wire signed [18:0] s1 = score_r[1*19 +: 19];
    wire signed [18:0] s2 = score_r[2*19 +: 19];
    wire signed [18:0] s3 = score_r[3*19 +: 19];

    assign cls_o = (s0 >= s1 && s0 >= s2 && s0 >= s3) ? 3'd0 :
                   (s1 >= s2 && s1 >= s3)               ? 3'd1 :
                   (s2 >= s3)                           ? 3'd2 : 3'd3;

endmodule


// =====================================================================================
// bnn_wblob -- 271 word weight blob -> engine registers
//
// Blob bit i -> word i/32, bit i%32  (LSB first inside the word; authoritative order
// of scripts/gen_weights.py).  Field map (weights_layout_spec.md v5 sec.2/sec.3):
//   [   0,  72) L1 weights   lane c tap t -> bit c*9+t
//   [  72, 144) L2 weights   same
//   [ 144, 208) L3 weights   out o in i  -> bit 144 + o*8+i
//   [ 208,8400) L5 weights   beat-major: bit 208+32q+8c+j = fc_w[c][8q+j]
//   [8400,8432) theta1 8 x 4      [8432,8464) theta2 8 x 4
//   [8464,8496) theta3 8 x 4      [8496,8552) theta4 8 x 7
//   [8552,8616) thetaFC 4 x 16 Q6 signed two's complement
//   [8616,8632) ID (0x0001)       [8632,8648) CRC16 (not checked here)
//
// 208 = 6*32 + 16, so every 32-bit beat still straddles two blob words:
//     beat q = { word(7+q)[15:0], word(6+q)[31:16] }        q = 0..255
// In v5 that beat IS the l5_fc weight word {b3,b2,b1,b0}, so the blob word arriving at
// address 7+q can be written straight to l5_fc (address q, data {d[15:0], fc_hi}) with
// no buffer, no second pass and no 256 x 32 register array.
// Verilog-2001, synthesizable.
module bnn_wblob (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        w_we,
    input  wire [8:0]  w_addr,
    input  wire [31:0] w_data,
    output reg  [71:0] l1_w,
    output reg  [71:0] l2_w,
    output reg  [63:0] w3,
    output reg  [31:0] th1_v,
    output reg  [31:0] th2_v,
    output reg  [31:0] th3_v,
    output reg  [55:0] th4_v,
    output reg  [63:0] thfc_v,
    // theta push (to theta_cmp / th3_r / th4_r / l5_fc)
    output reg         push_we,
    output reg  [2:0]  push_sel,     // 0=th1 1=th2 2=th3 3=th4 4=thFC
    output reg  [2:0]  push_ch,
    output reg  [15:0] push_data,
    // L5 weight write
    output reg         l5_we,
    output reg  [7:0]  l5_addr,
    output reg  [31:0] l5_data,
    output reg         load_done
);

    localparam [1:0] S_IDLE  = 2'd0,
                     S_BLOB  = 2'd1,
                     S_PUSH  = 2'd2,
                     S_READY = 2'd3;

    reg [1:0] st;

    // ---- beat word assembly: upper half of the previous blob word ----
    reg [15:0] fc_hi;

    // ---- theta push counter ----
    reg [5:0]  pcnt;                        // push counter 0..35

    // ================================================================ blob capture
    wire [8:0] a = w_addr;
    wire [31:0] d = w_data;
    wire       fc_hi_ld = (a >= 9'd6) && (a <= 9'd261);   // word 6+q upper half = beat q low
    wire       fc_wr    = (a >= 9'd7) && (a <= 9'd262);   // word 7+q lower half completes beat q
    wire [8:0] fc_beat  = a - 9'd7;                       // 0..255 = l5 beat / w_addr

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE;
            l1_w <= 72'd0; l2_w <= 72'd0; w3 <= 64'd0;
            th1_v <= 32'd0; th2_v <= 32'd0; th3_v <= 32'd0;
            th4_v <= 56'd0; thfc_v <= 64'd0;
            fc_hi <= 16'd0;
            pcnt <= 6'd0;
            l5_we <= 1'b0; l5_addr <= 8'd0; l5_data <= 32'd0;
            load_done <= 1'b0;
        end else begin
            l5_we     <= 1'b0;
            load_done <= 1'b0;

            // ---------------- weight blob capture ----------------
            if (w_we) begin
                case (a)
                    9'd0: l1_w[31:0]  <= d;
                    9'd1: l1_w[63:32] <= d;
                    9'd2: begin l1_w[71:64] <= d[7:0];  l2_w[23:0] <= d[31:8]; end
                    9'd3: l2_w[55:24] <= d;
                    9'd4: begin l2_w[71:56] <= d[15:0]; w3[15:0]   <= d[31:16]; end
                    9'd5: w3[47:16]  <= d;
                    9'd6: w3[63:48]  <= d[15:0];
                    // theta1/2/3 start at bit 8400 = word 262 bit 16
                    9'd262: th1_v[15:0]  <= d[31:16];
                    9'd263: begin th1_v[31:16] <= d[15:0];  th2_v[15:0]  <= d[31:16]; end
                    9'd264: begin th2_v[31:16] <= d[15:0];  th3_v[15:0]  <= d[31:16]; end
                    9'd265: begin th3_v[31:16] <= d[15:0];  th4_v[15:0]  <= d[31:16]; end
                    9'd266: begin th4_v[31:16] <= d[15:0];  th4_v[47:32] <= d[31:16]; end
                    9'd267: begin th4_v[55:48] <= d[7:0];
                                  thfc_v[15:0]  <= d[23:8];
                                  thfc_v[23:16] <= d[31:24]; end
                    9'd268: begin thfc_v[31:24] <= d[7:0];
                                  thfc_v[47:32] <= d[23:8];
                                  thfc_v[55:48] <= d[31:24]; end
                    9'd269: begin thfc_v[63:56] <= d[7:0];   // ID/CRC ignored here
                                  end
                    default: ;
                endcase
                // v5: beat q's low half arrives on word 6+q (remembered in fc_hi), the
                // high half on word 7+q -> the completed beat word goes straight to l5_fc.
                if (fc_hi_ld) fc_hi <= d[31:16];
                if (fc_wr) begin
                    l5_we   <= 1'b1;
                    l5_addr <= fc_beat[7:0];
                    l5_data <= {d[15:0], fc_hi};
                end
            end

            case (st)
            // ------------------------------------------------ wait for word 0
            S_IDLE: begin
                if (w_we && (a == 9'd0)) st <= S_BLOB;
            end
            // ------------------------------------------------ words 1..270 (FC streamed live)
            S_BLOB: begin
                if (w_we && (a == 9'd270)) begin
                    st   <= S_PUSH;
                    pcnt <= 6'd0;
                end
            end
            // ------------------------------------------------ theta push (36 cycles)
            S_PUSH: begin
                // push_we / push_sel / push_ch / push_data are combinational (below),
                // so the consumer samples index `pcnt` at the end of THIS cycle.
                if (pcnt == 6'd35) begin
                    load_done <= 1'b1;
                    st        <= S_READY;
                end else begin
                    pcnt <= pcnt + 6'd1;
                end
            end
            // ------------------------------------------------ done
            S_READY: begin
                // stays here; a second blob may restart the whole load
                if (w_we && (a == 9'd0)) st <= S_BLOB;
            end
            default: st <= S_IDLE;
            endcase
        end
    end

    // ---- theta push driver (combinational, indexed by pcnt) ----
    always @* begin
        push_we   = (st == S_PUSH);
        push_sel  = 3'd0;
        push_ch   = pcnt[2:0];
        push_data = 16'd0;
        if (st == S_PUSH) begin
            if (pcnt < 6'd8) begin
                push_sel  = 3'd0;
                push_data = {12'd0, th1_v[pcnt[2:0]*4 +: 4]};
            end else if (pcnt < 6'd16) begin
                push_sel  = 3'd1;
                push_data = {12'd0, th2_v[pcnt[2:0]*4 +: 4]};
            end else if (pcnt < 6'd24) begin
                push_sel  = 3'd2;
                push_data = {12'd0, th3_v[pcnt[2:0]*4 +: 4]};
            end else if (pcnt < 6'd32) begin
                push_sel  = 3'd3;
                push_data = {9'd0, th4_v[pcnt[2:0]*7 +: 7]};
            end else begin
                push_sel  = 3'd4;
                push_ch   = {1'b0, pcnt[1:0]};
                push_data = thfc_v[pcnt[1:0]*16 +: 16];
            end
        end
    end

endmodule
