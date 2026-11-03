// accum.v -- 8 parallel 12-bit accumulators (one per output channel / lane)
//
// Per channel c:
//   ld  && ld_ch==c        : acc <= ld_data              (priority 1)
//   en                     : acc <= (clr ? 0 : acc) + din[c]   (priority 2)
//   clr  (no en)           : acc <= 0                    (priority 3)
//   otherwise              : hold
//
// "clr + en together" is the normal 1-pixel/cycle case: it means "start a fresh
// accumulation with this cycle's lane result" (acc <= din), so the engine needs no
// separate clear cycle per pixel.  Without en, clr just zeroes the channel.
//
// Width: spec v2.2 6.1 fixes the accumulator at 12 bit for a max sum of 9 per pixel.
// There is no saturation; 12 bit holds 455 worst-case adds of 9, i.e. the register has
// to be cleared at the layer / pixel boundary (normal use).
// Verilog-2001, synthesizable, no DSP.
`timescale 1ns / 1ps

module accum #(
    parameter integer W = 12          // accumulator width
) (
    input  wire           clk,
    input  wire           rst_n,      // async active-low
    input  wire           clr,        // clear (with en: clear before this add)
    input  wire           ld,         // load ld_data into channel ld_ch
    input  wire [2:0]     ld_ch,      // channel select for ld (0..7)
    input  wire [W-1:0]   ld_data,    // load data
    input  wire           en,         // add the lane counts this cycle
    input  wire [31:0]    din,        // 8 x 4 lane match counts (0..9)
    output wire [8*W-1:0] acc         // channel c -> acc[c*W +: W]
);

    genvar ch;
    generate
        for (ch = 0; ch < 8; ch = ch + 1) begin : g_ch
            reg [W-1:0] r;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    r <= {W{1'b0}};
                else if (ld && (ld_ch == ch))                     // ld_ch zero-extends to ch
                    r <= ld_data;
                else if (en)
                    r <= (clr ? {W{1'b0}} : r) + din[ch*4 +: 4];  // 4-bit din zero-extends
                else if (clr)
                    r <= {W{1'b0}};
            end

            assign acc[ch*W +: W] = r;
        end
    endgenerate

endmodule
