// win_buf.v -- 3-row shift-register line buffer + 3x3 sliding window, 1 shift per cycle
//
// ================================= stream contract =================================
//  * `sof` (1 cycle) starts a layer: clears the three row stores and the position
//    counters.  Every layer must be preceded by exactly one sof.
//  * While `din_req`=1 the buffer wants one data pixel *this* cycle: drive `din`
//    (W bits, bit=1 -> +1) and `din_we`=1 in that same cycle.
//    din_req=1 & din_we=0 stalls the buffer (no shift, counters hold) -> back-pressure OK.
//  * The buffer inserts the SAME padding itself:
//       - one padding column (bit 0 == -1 in the +/-1 domain) after each 64 data columns,
//       - one padding row after the 64 data rows.
//    Layer length is therefore CW*CR = 65*65 = 4225 shift cycles for 4096 data pixels
//    (64*64).  din_req is high exactly 4096 times.
//  * `win` (combinational) is the 3x3 window centred on pixel (win_row, win_col);
//    `win_valid` (registered, 1 cycle) marks it.  Window bit order:
//       win[c*9 + ky*3 + kx],  ky=0 -> row above, kx=0 -> column left, so bit 0 is
//    the top-left tap and the weights of 05_权重布局 use exactly this order
//    ([out_ch][ky][kx], kx innermost).
//  * `done` pulses 1 cycle together with the last valid window (centre 63,63).
//
// ================================== why CW = 65 ===================================
//  With a pure shift-register cascade the row-to-row distance is exactly one row
//  length, so rows only align if the row store length equals the row stream length.
//  With a 64-bit row store (the literal "3 x 64bit" of spec 6.2) you get
//    (a) no column-64 slot -> the centres of column 63 can never be formed, and
//    (b) at column 0 the kx=0 tap of the rows above is the *previous row's last
//        pixel* instead of padding -> the left edge is wrong.
//  A 65th (always 0) padding bit per row fixes both, costs 3 FFs and +129 cycles per
//  layer (+3.15%), and keeps the result bit-exact with train_bnn10.matches_fixed().
//
//  Taps: s0 = row r (newest) / s1 = row r-1 / s2 = row r-2; the newest three bit
//  positions of each store are the columns col-2, col-1, col (col = incoming column),
//  i.e. the window of centre (row-1, col-1).  Verified against an array reference
//  model in tb_engine.v for W=1 and W=8.
//
// Verilog-2001, synthesizable, no BRAM (3 x CW x W flip-flops).
`timescale 1ns / 1ps

module win_buf #(
    parameter integer W  = 1,      // pixel width in bits = parallel planes (1 for L1, 8 for L2)
    parameter integer CW = 65,     // row store length = 64 data columns + 1 padding column
    parameter integer CR = 65      // rows per layer  = 64 data rows    + 1 padding row
) (
    input  wire           clk,
    input  wire           rst_n,      // async active-low
    input  wire           sof,        // start of layer (1 cycle)
    input  wire           din_we,     // pixel valid
    input  wire [W-1:0]   din,        // pixel (W planes, bit=1 -> +1)
    output wire [9*W-1:0] win,        // 3x3 window, lane c -> win[c*9 +: 9]
    output reg            win_valid,  // 1 -> win/win_row/win_col hold a valid window
    output reg  [5:0]     win_row,    // window centre row 0..63
    output reg  [5:0]     win_col,    // window centre col 0..63
    output wire           din_req,    // buffer wants a data pixel this cycle
    output reg            done        // layer finished (1 cycle)
);

    localparam integer NCW = W * CW;  // bits per row store

    reg [NCW-1:0] s0;                 // row r    (newest)
    reg [NCW-1:0] s1;                 // row r-1
    reg [NCW-1:0] s2;                 // row r-2

    reg [6:0] row;                    // 0..CR-1  (7 bit -> CW/CR must stay <= 128)
    reg [6:0] col;                    // 0..CW-1
    reg       active;                 // layer in progress

    // ---- control ----
    // data cycles need din_we, padding cycles (col==CW-1 or row==CR-1) always shift 0
    assign din_req = active && (row < (CR - 1)) && (col < (CW - 1));

    wire shift_en = active && (din_req ? din_we : 1'b1);
    wire [W-1:0] bit_in = (din_req && din_we) ? din : {W{1'b0}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0        <= {NCW{1'b0}};
            s1        <= {NCW{1'b0}};
            s2        <= {NCW{1'b0}};
            row       <= 7'd0;
            col       <= 7'd0;
            active    <= 1'b0;
            win_valid <= 1'b0;
            win_row   <= 6'd0;
            win_col   <= 6'd0;
            done      <= 1'b0;
        end else if (sof) begin
            s0        <= {NCW{1'b0}};
            s1        <= {NCW{1'b0}};
            s2        <= {NCW{1'b0}};
            row       <= 7'd0;
            col       <= 7'd0;
            active    <= 1'b1;
            win_valid <= 1'b0;
            done      <= 1'b0;
        end else begin
            done      <= 1'b0;
            win_valid <= 1'b0;

            if (shift_en) begin
                // cascade: the bit leaving s0 is the pixel one row earlier
                s0 <= {bit_in, s0[NCW-1:W]};
                s1 <= {s0[W-1:0], s1[NCW-1:W]};
                s2 <= {s1[W-1:0], s2[NCW-1:W]};

                // after this edge the stores hold the window of centre (row-1, col-1)
                win_valid <= (row != 7'd0) && (col != 7'd0);
                win_row   <= row[5:0] - 6'd1;      // row==0 is invalid, wrap value unused
                win_col   <= col[5:0] - 6'd1;

                // advance the stream position
                if (col == (CW - 1)) begin
                    col <= 7'd0;
                    if (row == (CR - 1)) begin
                        active <= 1'b0;
                        done   <= 1'b1;
                    end else begin
                        row <= row + 7'd1;
                    end
                end else begin
                    col <= col + 7'd1;
                end
            end
        end
    end

    // ---- 3x3 window taps: fixed wiring, no mux ----
    // Storage layout is pixel-interleaved: bit NCW-1-W*j-ch holds lane ch of the pixel
    // j positions before the incoming one, so the tap at column offset (kx-2) is
    //   NCW - W*(3-kx) + ch   ->  W=1 gives CW-3+kx, W=8 gives the top 8-bit slots.
    genvar gch, gky, gkx;
    generate
        for (gch = 0; gch < W; gch = gch + 1) begin : g_ch
            for (gky = 0; gky < 3; gky = gky + 1) begin : g_ky
                for (gkx = 0; gkx < 3; gkx = gkx + 1) begin : g_kx
                    assign win[gch*9 + gky*3 + gkx] =
                        (gky == 0) ? s2[NCW - W*(3 - gkx) + gch] :   // row above
                        (gky == 1) ? s1[NCW - W*(3 - gkx) + gch] :   // centre row
                                     s0[NCW - W*(3 - gkx) + gch];    // row below
                end
            end
        end
    endgenerate

endmodule
