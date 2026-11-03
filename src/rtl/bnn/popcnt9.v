// popcnt9.v -- 9-bit population count, pure combinational adder tree, 4-bit result (0..9)
//
// BNN MAC primitive: for one lane the 9 tap products are +1/-1, so the match count is
// popcount(XNOR(win, wgt)) with range 0..9 -> 4 bit (spec v2.2 6.1: accumulate range 0..9).
// Verilog-2001, synthesizable, no DSP, no BRAM.
`timescale 1ns / 1ps

module popcnt9 (
    input  wire [8:0] din,      // bits to count
    output wire [3:0] cnt       // number of ones in din, 0..9
);

    // ---- stage 1: three 3-bit groups -> 2-bit partial counts (0..3) ----
    wire [1:0] g0 = {1'b0, din[0]} + {1'b0, din[1]} + {1'b0, din[2]};
    wire [1:0] g1 = {1'b0, din[3]} + {1'b0, din[4]} + {1'b0, din[5]};
    wire [1:0] g2 = {1'b0, din[6]} + {1'b0, din[7]} + {1'b0, din[8]};

    // ---- stage 2: g0 + g1 -> 3-bit (0..6) ----
    wire [2:0] s01 = g0 + g1;

    // ---- stage 3: s01 + g2 -> 4-bit (0..9) ----
    assign cnt = s01 + g2;

endmodule
