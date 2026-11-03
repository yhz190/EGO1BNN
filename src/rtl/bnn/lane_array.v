// lane_array.v -- 8 lanes x 9 taps XNOR + popcount (the BNN MAC array)
//
// One lane = one output channel of a 3x3 binary convolution:
//   cnt[c] = popcount( ~(win[c] ^ wgt[c]) )  = number of matching +1/-1 products, 0..9
//   win bit = 1 means +1 (bright / hand), w bit = 1 means +1   (spec 05 sec.1)
//
// Packing (identical for win and wgt):
//   lane c occupies bits [c*9 +: 9]; inside a lane tap t = ky*3 + kx, t=0 = top-left.
//   L1 (1 plane, 8 output channels): fan the same 9-bit window out to all 8 lanes
//                                    -> win = {8{win9}}, wgt = per-channel 3x3 weights.
//   L2 (8 planes, depthwise)       : win_buf W=8 feeds lane c with plane c
//                                    -> each lane sees only its own plane's 9 taps.
// Verilog-2001, synthesizable, 72 XNOR + 8 popcount9, no DSP.
`timescale 1ns / 1ps

module lane_array (
    input  wire [71:0] win,     // 8 x 9 activation window bits
    input  wire [71:0] wgt,     // 8 x 9 weight bits
    output wire [31:0] cnt      // lane c -> cnt[c*4 +: 4]
);

    genvar ch;
    generate
        for (ch = 0; ch < 8; ch = ch + 1) begin : g_lane
            wire [8:0] eq;
            assign eq = ~(win[ch*9 +: 9] ^ wgt[ch*9 +: 9]);   // XNOR per tap
            popcnt9 u_pc (
                .din (eq),
                .cnt (cnt[ch*4 +: 4])
            );
        end
    endgenerate

endmodule
