// theta_cmp.v -- per-channel binarisation threshold compare
//
//   act[c] = (acc[c] > theta[c]) ? 1 : 0        (strictly greater, matches
//                                                train_bnn10.hw_eval: matches_fixed() > t)
// theta[c] is a W_TH-bit unsigned register (spec 05 sec.3: 4 bit, popcount units 0..9),
// loaded one channel per cycle through the th_we / th_ch / th_data port.
// The comparand is zero-extended to W_ACC bits; both operands are unsigned.
// th_val is a read-back port for debug / testbench use (synthesis drops it if unused).
// Verilog-2001, synthesizable.
`timescale 1ns / 1ps

module theta_cmp #(
    parameter integer W_ACC = 12,     // accumulator width
    parameter integer W_TH  = 4       // threshold width
) (
    input  wire              clk,
    input  wire              rst_n,   // async active-low
    input  wire              th_we,   // threshold write enable
    input  wire [2:0]        th_ch,   // threshold channel select
    input  wire [W_TH-1:0]   th_data, // threshold value
    input  wire [8*W_ACC-1:0] acc,    // 8 x W_ACC accumulator values
    output wire [7:0]        act,     // 8 activation bits (bit=1 -> +1)
    output wire [8*W_TH-1:0] th_val   // threshold read-back
);

    genvar ch;
    generate
        for (ch = 0; ch < 8; ch = ch + 1) begin : g_ch
            reg [W_TH-1:0] th;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    th <= {W_TH{1'b0}};
                else if (th_we && (th_ch == ch))
                    th <= th_data;
            end

            assign act[ch]    = (acc[ch*W_ACC +: W_ACC] > th);      // th zero-extends
            assign th_val[ch*W_TH +: W_TH] = th;
        end
    endgenerate

endmodule
