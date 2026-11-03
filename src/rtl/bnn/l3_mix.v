// l3_mix.v -- L3: 1x1 convolution 8 -> COUT (COUT x popcount8 + per-class theta compare)
//
// patch2 (v2.2-patch2, 2026-11-01): COUT = 8 (was 4).
// The output channels are fully PARALLEL (COUT x popcount8), so the layer length
// does not change: L3 is still exactly 4096 pixel beats for a 64x64 map.
//
// For one pixel (one beat) the input is the 8 plane bits of ACT_B and every
// output channel has 8 weight bits (spec 05 v4 sec.1/2: [out_ch][in_ch]):
//
//   cnt[k] = popcount8( ~(din ^ w3[k*8 +: 8]) )        k = 0..COUT-1, 0..8
//   act[k] = (cnt[k] > th3[k*W_TH +: W_TH]) ? 1 : 0    strict greater, == hw_eval
//
// K=8 so there is no padding and no 3x3 window: L3 is exactly 4096 beats.
// popcnt9 is reused with a leading 0 (din = {1'b0, eq}) so the result range is
// 0..8 and the block stays bit-identical with the verified primitive.
//
// The output is REGISTERED (one beat per cycle): en at cycle t -> act/valid at
// t+1.  layer_seq already tracks that one-cycle write delay, so the ACT write
// for beat n happens at cycle n+1 with waddr = n.
//
// Resource cost of the 4 -> 8 change: +4 popcount8 (~40 LUT), +4 theta compare
// (~8 LUT), +4 write bits.  No extra cycle.
// Verilog-2001, synthesizable, no DSP.
`timescale 1ns / 1ps

module l3_mix #(
    parameter integer COUT = 8,       // output channels (patch2: 8, patch1: 4)
    parameter integer W_TH = 4        // theta width (popcount units, 0..8)
) (
    input  wire                  clk,
    input  wire                  rst_n,    // async active-low
    input  wire                  en,       // 1 -> din is a valid pixel this cycle
    input  wire [7:0]            din,      // 8 input plane bits
    input  wire [8*COUT-1:0]     w3,       // class k weights in w3[k*8 +: 8]
    input  wire [COUT*W_TH-1:0]  th3,      // class k threshold in th3[k*W_TH +: W_TH]
    output reg  [COUT-1:0]       act,      // COUT activation bits (bit=1 -> +1)
    output reg  [COUT*4-1:0]     cnt,      // COUT x 4 bit match counts (TB / debug)
    output reg                   valid     // act/cnt valid (en delayed 1 cycle)
);

    wire [COUT*4-1:0] pc_all;              // class k count in pc_all[k*4 +: 4]
    genvar k;
    generate
        for (k = 0; k < COUT; k = k + 1) begin : g_pc
            wire [7:0] eq = ~(din ^ w3[k*8 +: 8]);
            popcnt9 u_pc (
                .din ({1'b0, eq}),
                .cnt (pc_all[k*4 +: 4])
            );
        end
    endgenerate

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act   <= {COUT{1'b0}};
            cnt   <= {(COUT*4){1'b0}};
            valid <= 1'b0;
        end else begin
            valid <= 1'b0;
            if (en) begin
                for (i = 0; i < COUT; i = i + 1) begin
                    cnt[i*4 +: 4] <= pc_all[i*4 +: 4];
                    act[i]        <= (pc_all[i*4 +: 4] > th3[i*W_TH +: W_TH]);
                end
                valid <= 1'b1;
            end
        end
    end

endmodule
