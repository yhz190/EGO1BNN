// ============================================================================
// vga_simple.v -- EGO1 VGA generator for T5 (640x480@60, 12-bit 4:4:4)
// ----------------------------------------------------------------------------
// Goal: a complete self-test picture plus the classification result overlay
// without any CPU / UART interaction ("pull the serial cable and it still
// displays").  Everything drawn here comes from the raster counters plus a few
// result registers, so the module works from power-on with idle inputs.
//
// Memory: NO framebuffer and NO BRAM at all.  Each pixel is produced
// combinationally from (x, y) and the result registers, then registered once on
// the 25 MHz pixel tick.  The only "font storage" is a case-based constant mux
// (21 glyphs x 8 byte = 1344 bit) that synthesises into LUTs, not a RAM.
//
// Picture layout (640 x 480, origin top-left):
//   y   0.. 39  title colour bar, 8 bars x 80 px (the power-on self-test pattern)
//   y  40.. 41  separator line
//   y  96..127  class name, 8x8 dot-matrix font x4 -> 32x32 per char, centred
//               ROCK / PAPER / SCISSORS / JUNK, colour coded per class
//   y 200..331  four score bars (h=24, pitch 36, track x 80..591 = 512 px)
//               with a 0..3 digit label at x 24..39 (8x8 font x2)
//   y 432..463  status word bottom-left, 8x8 font x4: IDLE / BUSY / DONE / ERR
//   everything else: dark navy background
//
// Score bars: score[k] is 19-bit signed Q6.  len = clamp((score>>>6) + 256, 0, 512),
// i.e. a score of 0 fills half the track, positive scores are longer, negative
// shorter.  The mapping is monotonic in the signed score and saturates at both
// ends instead of wrapping.
//
// Timing (25.0 MHz pixel clock -> 59.52 Hz frame rate)
//   H: 640 visible + 16 FP + 96 sync + 48 BP = 800 pixels = 32.000 us
//   V: 480 visible + 10 FP +  2 sync + 33 BP = 525 lines  = 16.800 ms
//   hs/vs are negative polarity (low during sync), the standard 640x480@60 mode.
//
// Pipeline: hcnt/vcnt/rgb/hs/vs are all registered on the pixel tick and all
// describe the SAME pixel (colour and sync computed from the next counter value),
// so there is no skew between the picture and the sync edges.
//
// Verilog-2001, synthesizable, no BRAM, no DSP.
// ============================================================================
`timescale 1ns / 1ps

module vga_simple (
    input  wire        clk,           // 50 MHz system clock (top_bnn: BUFG 50 MHz)
    input  wire        rst_n,         // asynchronous reset, active low
    input  wire        pix_clk_en,    // pixel engine enable: 1 = raster runs
                                      // (tie 1'b1 in top_bnn; 0 freezes the raster)
    input  wire [2:0]  cls,           // 0=rock 1=paper 2=scissors 3=junk
    input  wire [75:0] score,         // class k in score[k*19 +: 19], signed Q6
    input  wire        busy,          // engine busy
    input  wire        done,          // run finished (pulse or level -> latched)
    input  wire        err,           // any error flag (sticky in the SFR)
    output reg         vga_hs,
    output reg         vga_vs,
    output reg  [11:0] vga_rgb
);

    // ------------------------------------------------------------ timing (25 MHz)
    localparam [9:0] H_VIS   = 10'd640;
    localparam [9:0] H_TOTAL = 10'd800;
    localparam [9:0] H_SYNC0 = 10'd656;      // 640 + 16 front porch
    localparam [9:0] H_SYNC1 = 10'd752;      // + 96 sync  -> 48 back porch
    localparam [9:0] V_VIS   = 10'd480;
    localparam [9:0] V_TOTAL = 10'd525;
    localparam [9:0] V_SYNC0 = 10'd490;      // 480 + 10 front porch
    localparam [9:0] V_SYNC1 = 10'd492;      // +  2 sync  -> 33 back porch

    // ------------------------------------------------------------ layout
    localparam [9:0] TITLE_H = 10'd40;       // y 0..39    colour bars
    localparam [9:0] SEP_Y   = 10'd42;       // y 40..41   separator
    localparam [9:0] TXT_Y   = 10'd96;       // y 96..127  class name (x4)
    localparam [9:0] TXT_H   = 10'd32;
    localparam [9:0] BAR_Y   = 10'd200;      // first score bar
    localparam [9:0] BAR_P   = 10'd36;       // bar pitch
    localparam [9:0] BAR_H   = 10'd24;       // bar height
    localparam [9:0] BAR_X   = 10'd80;       // bar track x
    localparam [9:0] BAR_MAX = 10'd512;      // bar track length
    localparam [9:0] LBL_X   = 10'd24;       // digit label x
    localparam [9:0] STA_Y   = 10'd432;      // y 432..463 status word (x4)
    localparam [9:0] STA_X   = 10'd16;

    // ------------------------------------------------------------ colours (4:4:4)
    localparam [11:0] C_BG   = 12'h024;      // dark navy background
    localparam [11:0] C_SEP  = 12'h444;      // separator line
    localparam [11:0] C_TRK  = 12'h222;      // score-bar track
    localparam [11:0] C_LBL  = 12'hCCC;      // digit label

    // ------------------------------------------------------------ score -> bar length
    // score[k] = 64 * (2*acc[k] - NF) + bias  (Q6), so >>6 is the integer score.
    // +256 puts score 0 at half track, then saturate to [0, 512].
    wire signed [18:0] sc0 = score[ 0*19 +: 19];
    wire signed [18:0] sc1 = score[ 1*19 +: 19];
    wire signed [18:0] sc2 = score[ 2*19 +: 19];
    wire signed [18:0] sc3 = score[ 3*19 +: 19];

    wire signed [12:0] si0 = sc0 >>> 6;
    wire signed [12:0] si1 = sc1 >>> 6;
    wire signed [12:0] si2 = sc2 >>> 6;
    wire signed [12:0] si3 = sc3 >>> 6;

    wire signed [14:0] of0 = si0 + 15'sd256;
    wire signed [14:0] of1 = si1 + 15'sd256;
    wire signed [14:0] of2 = si2 + 15'sd256;
    wire signed [14:0] of3 = si3 + 15'sd256;

    wire [9:0] len0 = (of0 <= 15'sd0) ? 10'd0 : (of0 >= 15'sd512) ? BAR_MAX : of0[9:0];
    wire [9:0] len1 = (of1 <= 15'sd0) ? 10'd0 : (of1 >= 15'sd512) ? BAR_MAX : of1[9:0];
    wire [9:0] len2 = (of2 <= 15'sd0) ? 10'd0 : (of2 >= 15'sd512) ? BAR_MAX : of2[9:0];
    wire [9:0] len3 = (of3 <= 15'sd0) ? 10'd0 : (of3 >= 15'sd512) ? BAR_MAX : of3[9:0];

    // ------------------------------------------------------------ class / status
    // class codes outside 0..3 are shown as JUNK instead of an undefined glyph
    wire [1:0] cls_i = cls[2] ? 2'd3 : cls[1:0];

    reg  done_seen;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)    done_seen <= 1'b0;
        else if (busy) done_seen <= 1'b0;      // a new run clears DONE
        else if (done) done_seen <= 1'b1;      // 1-cycle pulse is latched here
    end

    // BUSY wins while a run is in progress (an old ERR must not hide it)
    wire [1:0] st = busy ? 2'd1 : err ? 2'd3 : done_seen ? 2'd2 : 2'd0;
    //                0=IDLE      1=BUSY      2=DONE        3=ERR

    // ============================================================ character ROM
    // 21 glyphs, 8x8, row 0 is the top row, bit 7 is the leftmost pixel.
    // 21 x 64 bit = 1344 bit -> constant mux in LUTs (no BRAM).
    //  1 R   2 O   3 C   4 K   5 P   6 A   7 E   8 S   9 I  10 J  11 U
    // 12 N  13 D  14 L  15 B  16 Y  17 '0' 18 '1' 19 '2' 20 '3'
    function [63:0] glyph;
        input [4:0] ch;
        begin
            case (ch)
                5'd1:  glyph = {8'h7C,8'h42,8'h42,8'h7C,8'h48,8'h44,8'h42,8'h00}; // R
                5'd2:  glyph = {8'h3C,8'h42,8'h42,8'h42,8'h42,8'h42,8'h3C,8'h00}; // O
                5'd3:  glyph = {8'h3C,8'h42,8'h40,8'h40,8'h40,8'h42,8'h3C,8'h00}; // C
                5'd4:  glyph = {8'h42,8'h44,8'h48,8'h70,8'h48,8'h44,8'h42,8'h00}; // K
                5'd5:  glyph = {8'h7C,8'h42,8'h42,8'h7C,8'h40,8'h40,8'h40,8'h00}; // P
                5'd6:  glyph = {8'h3C,8'h42,8'h42,8'h7E,8'h42,8'h42,8'h42,8'h00}; // A
                5'd7:  glyph = {8'h7E,8'h40,8'h40,8'h7C,8'h40,8'h40,8'h7E,8'h00}; // E
                5'd8:  glyph = {8'h3E,8'h40,8'h40,8'h3C,8'h02,8'h02,8'h7C,8'h00}; // S
                5'd9:  glyph = {8'h7C,8'h10,8'h10,8'h10,8'h10,8'h10,8'h7C,8'h00}; // I
                5'd10: glyph = {8'h1F,8'h04,8'h04,8'h04,8'h04,8'h44,8'h38,8'h00}; // J
                5'd11: glyph = {8'h42,8'h42,8'h42,8'h42,8'h42,8'h42,8'h3C,8'h00}; // U
                5'd12: glyph = {8'h42,8'h62,8'h52,8'h4A,8'h46,8'h42,8'h42,8'h00}; // N
                5'd13: glyph = {8'h78,8'h44,8'h42,8'h42,8'h42,8'h44,8'h78,8'h00}; // D
                5'd14: glyph = {8'h40,8'h40,8'h40,8'h40,8'h40,8'h40,8'h7E,8'h00}; // L
                5'd15: glyph = {8'h7C,8'h42,8'h42,8'h7C,8'h42,8'h42,8'h7C,8'h00}; // B
                5'd16: glyph = {8'h42,8'h42,8'h24,8'h18,8'h18,8'h18,8'h18,8'h00}; // Y
                5'd17: glyph = {8'h3C,8'h46,8'h4A,8'h52,8'h62,8'h42,8'h3C,8'h00}; // 0
                5'd18: glyph = {8'h18,8'h28,8'h08,8'h08,8'h08,8'h08,8'h3E,8'h00}; // 1
                5'd19: glyph = {8'h3C,8'h42,8'h02,8'h04,8'h08,8'h10,8'h7E,8'h00}; // 2
                5'd20: glyph = {8'h3C,8'h42,8'h02,8'h1C,8'h02,8'h42,8'h3C,8'h00}; // 3
                default: glyph = 64'h0;                                              // blank
            endcase
        end
    endfunction

    // one row (0..7) of a glyph, bit 7 = leftmost pixel
    function [7:0] glyph_row;
        input [4:0] ch;
        input [2:0] row;
        reg   [63:0] g;
        begin
            g = glyph(ch);
            glyph_row = g[(7 - row)*8 +: 8];   // row 0 is the MSB byte
        end
    endfunction

    // ---- per-class colours, geometry and strings (class_map.py order) ----
    function [11:0] cls_col;
        input [1:0] c;
        begin
            case (c)
                2'd0:    cls_col = 12'hF00;    // rock     red
                2'd1:    cls_col = 12'h0F0;    // paper    green
                2'd2:    cls_col = 12'h0FF;    // scissors cyan
                default: cls_col = 12'hFA0;    // junk     amber
            endcase
        end
    endfunction

    function [9:0] cls_x0;                     // centred text origin
        input [1:0] c;
        begin
            case (c)
                2'd0:    cls_x0 = 10'd256;     // "ROCK"     4 chars
                2'd1:    cls_x0 = 10'd240;     // "PAPER"    5 chars
                2'd2:    cls_x0 = 10'd192;     // "SCISSORS" 8 chars
                default: cls_x0 = 10'd256;     // "JUNK"     4 chars
            endcase
        end
    endfunction

    function [3:0] cls_nch;
        input [1:0] c;
        begin
            case (c)
                2'd0:    cls_nch = 4'd4;
                2'd1:    cls_nch = 4'd5;
                2'd2:    cls_nch = 4'd8;
                default: cls_nch = 4'd4;
            endcase
        end
    endfunction

    function [4:0] cls_char;
        input [1:0] c;
        input [2:0] i;
        begin
            case (c)
                2'd0: case (i)          // R O C K
                        3'd0: cls_char = 5'd1;  3'd1: cls_char = 5'd2;
                        3'd2: cls_char = 5'd3;  default: cls_char = 5'd4;
                      endcase
                2'd1: case (i)          // P A P E R
                        3'd0: cls_char = 5'd5;  3'd1: cls_char = 5'd6;
                        3'd2: cls_char = 5'd5;  3'd3: cls_char = 5'd7;
                        default: cls_char = 5'd1;
                      endcase
                2'd2: case (i)          // S C I S S O R S
                        3'd0: cls_char = 5'd8;  3'd1: cls_char = 5'd3;
                        3'd2: cls_char = 5'd9;  3'd3: cls_char = 5'd8;
                        3'd4: cls_char = 5'd8;  3'd5: cls_char = 5'd2;
                        3'd6: cls_char = 5'd1;  default: cls_char = 5'd8;
                      endcase
                default: case (i)       // J U N K
                        3'd0: cls_char = 5'd10; 3'd1: cls_char = 5'd11;
                        3'd2: cls_char = 5'd12; default: cls_char = 5'd4;
                      endcase
            endcase
        end
    endfunction

    function [11:0] sta_col;
        input [1:0] s;
        begin
            case (s)
                2'd1:    sta_col = 12'hFF0;    // BUSY  yellow
                2'd2:    sta_col = 12'h0F0;    // DONE  green
                2'd3:    sta_col = 12'hF00;    // ERR   red
                default: sta_col = 12'h888;    // IDLE  grey
            endcase
        end
    endfunction

    function [3:0] sta_nch;
        input [1:0] s;
        begin
            sta_nch = (s == 2'd3) ? 4'd3 : 4'd4;   // "ERR" is 3 chars
        end
    endfunction

    function [4:0] sta_char;
        input [1:0] s;
        input [2:0] i;
        begin
            case (s)
                2'd1: case (i)          // B U S Y
                        3'd0: sta_char = 5'd15; 3'd1: sta_char = 5'd11;
                        3'd2: sta_char = 5'd8;  default: sta_char = 5'd16;
                      endcase
                2'd2: case (i)          // D O N E
                        3'd0: sta_char = 5'd13; 3'd1: sta_char = 5'd2;
                        3'd2: sta_char = 5'd12; default: sta_char = 5'd7;
                      endcase
                2'd3: case (i)          // E R R
                        3'd0: sta_char = 5'd7;  3'd1: sta_char = 5'd1;
                        default: sta_char = 5'd1;
                      endcase
                default: case (i)       // I D L E
                        3'd0: sta_char = 5'd9;  3'd1: sta_char = 5'd13;
                        3'd2: sta_char = 5'd14; default: sta_char = 5'd7;
                      endcase
            endcase
        end
    endfunction

    // ---- title colour bars: 8 x 80 px, exact integer boundaries ----
    function [11:0] bar_col;
        input [9:0] x;
        reg   [3:0] i;
        begin
            i = 4'd0;
            if (x >= 10'd80)  i = i + 4'd1;
            if (x >= 10'd160) i = i + 4'd1;
            if (x >= 10'd240) i = i + 4'd1;
            if (x >= 10'd320) i = i + 4'd1;
            if (x >= 10'd400) i = i + 4'd1;
            if (x >= 10'd480) i = i + 4'd1;
            if (x >= 10'd560) i = i + 4'd1;
            case (i)
                4'd0:    bar_col = 12'hFFF;    // white
                4'd1:    bar_col = 12'hFF0;    // yellow
                4'd2:    bar_col = 12'h0FF;    // cyan
                4'd3:    bar_col = 12'h0F0;    // green
                4'd4:    bar_col = 12'hF0F;    // magenta
                4'd5:    bar_col = 12'hF00;    // red
                4'd6:    bar_col = 12'h00F;    // blue
                default: bar_col = 12'h000;    // black
            endcase
        end
    endfunction

    // ---- bar length of bar i ----
    function [9:0] bar_len;
        input [1:0] i;
        begin
            case (i)
                2'd0:    bar_len = len0;
                2'd1:    bar_len = len1;
                2'd2:    bar_len = len2;
                default: bar_len = len3;
            endcase
        end
    endfunction

    // ============================================================ pixel generator
    // Pure combinational function of (x, y) and the result registers.  x/y are
    // always inside the visible area when this is called.
    function [11:0] pixgen;
        input [9:0] x;
        input [9:0] y;
        reg   [9:0] x0, rel, by;
        reg   [3:0] nch;
        reg   [2:0] ci;                      // 0..7: SCISSORS needs 8 chars
        reg   [1:0] bi;
        reg   [2:0] row, col;
        reg   [4:0] ch;
        reg   [7:0] bits;
        begin
            pixgen = C_BG;

            // ---- 1. title colour bars ----
            if (y < TITLE_H) begin
                pixgen = bar_col(x);

            // ---- 2. separator ----
            end else if (y < SEP_Y) begin
                pixgen = C_SEP;

            // ---- 3. class name (8x8 font, scale 4) ----
            end else if ((y >= TXT_Y) && (y < (TXT_Y + TXT_H))) begin
                x0  = cls_x0(cls_i);
                nch = cls_nch(cls_i);
                rel = x - x0;
                if ((x >= x0) && (rel < {nch, 5'b0})) begin
                    ci   = rel[8:5];                 // /32  -> which character
                    col  = rel[4:2];                 // /4 %8-> which font column
                    row  = (y - TXT_Y) >> 2;         // /4 %8-> which font row
                    ch   = cls_char(cls_i, ci);
                    bits = glyph_row(ch, row);
                    if (bits[7 - col]) pixgen = cls_col(cls_i);
                end

            // ---- 4. four score bars ----
            end else if ((y >= BAR_Y) && (y < (BAR_Y + 4*BAR_P))) begin
                by = y - BAR_Y;                      // 0 .. 143
                if (by < BAR_P - 10'd12) begin             // 0..23
                    bi = 2'd0;
                end else if ((by >= BAR_P) && (by < 2*BAR_P - 10'd12)) begin  // 36..59
                    bi = 2'd1; by = by - BAR_P;
                end else if ((by >= 2*BAR_P) && (by < 3*BAR_P - 10'd12)) begin // 72..95
                    bi = 2'd2; by = by - 2*BAR_P;
                end else if ((by >= 3*BAR_P) && (by < 4*BAR_P - 10'd12)) begin // 108..131
                    bi = 2'd3; by = by - 3*BAR_P;
                end else begin
                    bi = 2'd0; by = 10'd255;         // gap between bars
                end
                if (by != 10'd255) begin
                    rel = x - LBL_X;
                    // digit label: 8x8 font at scale 2 -> 16x16, x 24..39
                    if ((x >= LBL_X) && (x < (LBL_X + 10'd16)) &&
                        (by >= 10'd4) && (by < 10'd20)) begin
                        col  = rel[3:1];             // /2 % 8
                        row  = (by - 10'd4) >> 1;    // /2 % 8
                        ch   = 5'd17 + {3'd0, bi};   // '0'..'3'
                        bits = glyph_row(ch, row);
                        if (bits[7 - col]) pixgen = C_LBL;
                    end
                    // track + fill
                    else if ((x >= BAR_X) && (x < (BAR_X + BAR_MAX))) begin
                        rel = x - BAR_X;
                        pixgen = (rel < bar_len(bi)) ? cls_col(bi) : C_TRK;
                    end
                end

            // ---- 5. status word bottom-left (8x8 font, scale 4) ----
            end else if ((y >= STA_Y) && (y < (STA_Y + 10'd32))) begin
                nch = sta_nch(st);
                rel = x - STA_X;
                if ((x >= STA_X) && (rel < {nch, 5'b0})) begin
                    ci   = rel[8:5];
                    col  = rel[4:2];
                    row  = (y - STA_Y) >> 2;
                    ch   = sta_char(st, ci);
                    bits = glyph_row(ch, row);
                    if (bits[7 - col]) pixgen = sta_col(st);
                end
            end
        end
    endfunction

    // ============================================================ raster timing
    // 25 MHz pixel tick = /2 of the 50 MHz clock, gated by pix_clk_en.
    reg pdiv;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) pdiv <= 1'b0;
        else        pdiv <= ~pdiv;
    end
    wire pen = pix_clk_en & pdiv;

    reg  [9:0] hcnt, vcnt;

    wire hwrap = (hcnt == (H_TOTAL - 10'd1));
    wire [9:0] hnx = hwrap ? 10'd0 : (hcnt + 10'd1);
    wire [9:0] vnx = hwrap ? ((vcnt == (V_TOTAL - 10'd1)) ? 10'd0 : (vcnt + 10'd1))
                           : vcnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hcnt    <= 10'd0;
            vcnt    <= 10'd0;
            vga_hs  <= 1'b1;
            vga_vs  <= 1'b1;
            vga_rgb <= 12'h000;
        end else if (pen) begin
            hcnt    <= hnx;
            vcnt    <= vnx;
            vga_hs  <= ~((hnx >= H_SYNC0) && (hnx < H_SYNC1));
            vga_vs  <= ~((vnx >= V_SYNC0) && (vnx < V_SYNC1));
            vga_rgb <= ((hnx < H_VIS) && (vnx < V_VIS)) ? pixgen(hnx, vnx)
                                                        : 12'h000;
        end
    end

endmodule
