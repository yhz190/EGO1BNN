// l5_fc.v -- L5: fully connected NF -> 4, XNOR + popcount, Q6 fixed-point bias
//
// patch2 (v2.2-patch2, 2026-11-01): NF = 2048 (was 1024) and the per-class bias
// is now Q6 fixed point, 16 bit signed (was a 9 bit integer):
//
//   acc[k]   = sum over NF features of (feat[f] == wf[k][f])        0..NF   (12 bit)
//   score[k] = 64 * (2*acc[k] - NF) + bias_q[k]                    signed 19 bit
//   bias_q   = round(bias_float * 64)                              16 bit signed
//   pred     = argmax(score)
//
// Semantics = train_bnn10.hw_eval_q6() (the E6 定案 of sim/20261101_E6定案_Q6定点偏置.md).
// The old 9-bit integer bias could not separate two classes whose match counts
// are equal and whose float bias differs by less than 1 LSB; Q6 keeps 1/64
// resolution and is the accepted 口径 (int9 argmax mismatch 6/6 ckpt, Q6 0/6).
//
// Range (NF = 2048):  2*acc-NF in [-2048, 2048] -> x64 in [-131072, 131072]
//                     + bias_q in [-32768, 32767] -> [-163840, 163839]  ->  SW = 19
//
// Data flow: 8 features per beat, 4 classes in parallel (4 x popcount8), one
// accumulator per class.  NF/8 = 256 beats (+1 for the output register) = 257.
//   beat q: feat bits = features q*8 .. q*8+7, channel c = q/32, blocks
//           (q%32)*8 .. +7  (see l4_pool for why that is the PyTorch order)
//   weight word q = 4 classes x 8 feature bits, bit k*8+j = wf[k][q*8+j]
//   the FC weight blob (class-major, NF/8 bytes per class) loads one 32 bit word
//   per cycle: w_data = {blob3[q], blob2[q], blob1[q], blob0[q]}
//
// Output is REGISTERED: valid pulses 1 cycle after the last beat with
// score/acc_out stable.  Total L5 = NF/8 beats (+1 for the output register).
// Verilog-2001, synthesizable, no DSP, no multiplier (x64 is a wire shift).
`timescale 1ns / 1ps

module l5_fc #(
    parameter integer NF   = 2048,    // features (patch2: 2048, patch1: 1024)
    parameter integer NC   = 4,       // classes
    parameter integer NCH  = 8,       // feature channels of the L4 pool
    parameter integer W_TH = 16,      // signed Q6 bias width (patch2: 16)
    parameter integer SW   = 19       // signed score width (patch2: 19)
) (
    clk, rst_n, start, feat8, feat_addr, feat_ch,
    w_we, w_addr, w_data, th_we, th_ch, th_data,
    valid, score, acc_out
);

    // ---- derived widths ----
    function integer clog2f;
        input integer v;
        integer i;
        begin
            clog2f = 0;
            for (i = v - 1; i > 0; i = i >> 1) clog2f = clog2f + 1;
        end
    endfunction

    localparam integer NB  = NF / 8;            // weight words = beats (256)
    localparam integer QW  = clog2f(NB);        // beat index / weight address width (8)
    localparam integer WPC = NF / (8 * NCH);    // packed words per feature channel (32)
    localparam integer AW  = clog2f(WPC);       // feat_addr width (5)
    localparam integer CW  = clog2f(NCH);       // feat_ch   width (3)
    localparam integer ACW = clog2f(NF + 1);    // accumulator width, 0..NF (12)
    localparam integer DW  = ACW + 3;           // 2*acc - NF domain: 2*NF = 4096 needs 14 bit
                                                // unsigned / 15 bit signed headroom (15)

    input  wire                clk;
    input  wire                rst_n;       // async active-low
    input  wire                start;       // 1-cycle pulse: begin a NF/8-beat pass
    input  wire [7:0]          feat8;       // 8 feature bits of the current beat
    output wire [AW-1:0]       feat_addr;   // beat%WPC -> l4_pool.frd_addr
    output wire [CW-1:0]       feat_ch;     // beat/WPC -> l4_pool.frd_ch
    // ---- weight load: NF/8 words x 32 bit, one word per cycle ----
    input  wire                w_we;
    input  wire [QW-1:0]       w_addr;
    input  wire [31:0]         w_data;
    // ---- bias load: one class per cycle, two's complement Q6 ----
    input  wire                th_we;
    input  wire [1:0]          th_ch;
    input  wire [W_TH-1:0]     th_data;
    // ---- result ----
    output reg                 valid;       // 1 cycle: score/acc_out valid
    output reg  [NC*SW-1:0]    score;       // class k in score[k*SW +: SW]  (signed)
    output reg  [NC*ACW-1:0]   acc_out;     // class k in acc_out[k*ACW +: ACW] (0..NF)

    reg  [QW-1:0]  q;                       // beat index 0..NB-1
    reg            busy;
    reg  [31:0]    wmem [0:NB-1];
    reg  signed [W_TH-1:0] th [0:NC-1];
    reg  [ACW-1:0] acc [0:NC-1];            // per-class match counter 0..NF

    // The start cycle itself is beat 0.  q_eff forces the address to 0 on that
    // cycle: without it a *restart* would still see the last beat index left in
    // q (NB-1) and fold the wrong feature/weight word into acc -- caught by the
    // restart test in tb_pipeline.v.
    wire [QW-1:0] q_eff = (start && !busy) ? {QW{1'b0}} : q;
    wire [31:0]   wq    = wmem[q_eff];

    // ---- weight load port ----
    always @(posedge clk) begin
        if (w_we) wmem[w_addr] <= w_data;
    end

    // ---- bias load port ----
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < NC; k = k + 1) th[k] <= {W_TH{1'b0}};
        end else if (th_we) begin
            th[th_ch] <= th_data;
        end
    end

    // ---- 4 x popcount8 of the current beat ----
    wire [15:0] pc_all;                     // class k count in pc_all[k*4 +: 4]
    genvar g;
    generate
        for (g = 0; g < 4; g = g + 1) begin : g_pc
            wire [7:0] eq = ~(feat8 ^ wq[g*8 +: 8]);
            popcnt9 u_pc (
                .din ({1'b0, eq}),
                .cnt (pc_all[g*4 +: 4])
            );
        end
    endgenerate

    // ---- running sum including the current beat (0..NF) ----
    wire [NC*ACW-1:0] nx;
    assign nx[0*ACW +: ACW] = acc[0] + {{(ACW-4){1'b0}}, pc_all[3:0]};
    assign nx[1*ACW +: ACW] = acc[1] + {{(ACW-4){1'b0}}, pc_all[7:4]};
    assign nx[2*ACW +: ACW] = acc[2] + {{(ACW-4){1'b0}}, pc_all[11:8]};
    assign nx[3*ACW +: ACW] = acc[3] + {{(ACW-4){1'b0}}, pc_all[15:12]};

    // ---- score = 64 * (2*acc - NF) + bias_q   (signed, SW bit) ----
    localparam [DW-1:0] NFLOC = NF;         // NF zero-extended to the DW domain
    wire signed [SW-1:0] sc_w [0:NC-1];     // per-class score
    generate
        for (g = 0; g < NC; g = g + 1) begin : g_sc
            wire [DW-1:0]        twox = {1'b0, nx[g*ACW +: ACW], 1'b0};   // 2*acc, 0..2NF
            wire signed [DW-1:0] dd   = $signed(twox) - $signed(NFLOC);   // -NF .. NF
            wire signed [SW-1:0] d19  = {{(SW-DW){dd[DW-1]}}, dd};
            wire signed [SW-1:0] b19  = {{(SW-W_TH){th[g][W_TH-1]}}, th[g]};
            assign sc_w[g] = (d19 <<< 6) + b19;
        end
    endgenerate

    assign feat_addr = q_eff[AW-1:0];
    assign feat_ch   = q_eff[QW-1:AW];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q       <= {QW{1'b0}};
            busy    <= 1'b0;
            valid   <= 1'b0;
            score   <= {(NC*SW){1'b0}};
            acc_out <= {(NC*ACW){1'b0}};
            for (k = 0; k < NC; k = k + 1) acc[k] <= {ACW{1'b0}};
        end else begin
            valid <= 1'b0;
            if (start && !busy) begin
                // the start cycle itself is beat 0
                busy <= 1'b1;
                q    <= {{(QW-1){1'b0}}, 1'b1};
                for (k = 0; k < NC; k = k + 1)
                    acc[k] <= {{(ACW-4){1'b0}}, pc_all[k*4 +: 4]};
            end else if (busy) begin
                if (q == (NB - 1)) begin
                    busy <= 1'b0;
                    valid <= 1'b1;
                    for (k = 0; k < NC; k = k + 1) begin
                        acc[k]              <= nx[k*ACW +: ACW];
                        acc_out[k*ACW +: ACW] <= nx[k*ACW +: ACW];
                        score[k*SW +: SW]   <= sc_w[k];
                    end
                end else begin
                    q <= q + {{(QW-1){1'b0}}, 1'b1};
                    for (k = 0; k < NC; k = k + 1) acc[k] <= nx[k*ACW +: ACW];
                end
            end
        end
    end

endmodule
