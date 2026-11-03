// l4_pool.v -- L4: 4x4 block pooling + majority vote (>= TH / 16), NCH channels
//
// patch2 (v2.2-patch2, 2026-11-01): NCH = 8 (was 4).  The pooling geometry is
// UNCHANGED (already 4x4 = 16 pixels per block, majority >= TH/16 with TH = 8);
// only the channel count and the packed word width change:
//
//   patch1: 4 ch x 256 blocks                = 1024 feature bits, 32 words x 32 bit
//   patch2: 8 ch x 256 blocks                = 2048 feature bits, 32 words x 64 bit
//
// One pixel (NCH channel bits) per 'en' pulse.  16 pulses make one block:
//
//   sum[c]  = number of 1s of channel c inside the 4x4 block      (0..16)
//   feat[c] = (sum[c] >= th4[c*W_TH +: W_TH]) ? 1 : 0             (default 8 -> >=8/16)
//
// The block index is NOT decoded from pixel coordinates: the scan order is
// block-major (see layer_seq S_L4: 16 consecutive pixels of the same block),
// so the module only needs a 4 bit counter.  That is why 4096 beats are enough
// and no 16-deep per-block accumulator array is required.
//
// Output of a finished block: feat_we pulses 1 cycle after the 16th pixel with
// feat/feat_blk valid.  Features are packed into one word = 8 blocks x NCH
// channels:  word bits [j*NCH + c] = block (8*w + j), channel c.
// A full word is written every 8 blocks (no partial-word write needed), so the
// store is NW = NF/8 words of NCH*8 bit (32 x 64 bit = 2048 bit of distributed RAM).
//
// L5 reads the packed word (async distributed RAM) and selects one channel:
//   frd_data[j] = block (8*frd_addr + j), channel frd_ch
// which is exactly the PyTorch flatten order f = c*256 + by*16 + bx, because
// for beat q of L5 the channel is q/32 and the 8 blocks are (q%32)*8 .. +7.
//
// Cost of the 4 -> 8 change: +4 five-bit counters + 4 comparators (~80 LUT),
// packing register 32 -> 64 bit, feature RAM 1024 -> 2048 bit.  No extra cycle.
// Verilog-2001, synthesizable, no DSP, no multiplier.
`timescale 1ns / 1ps

module l4_pool #(
    parameter integer NCH  = 8,       // feature channels (patch2: 8, patch1: 4)
    parameter integer NF   = 256,     // blocks per channel (16x16 blocks of 4x4 pixels)
    parameter integer W_TH = 5        // threshold width (0..16 needs 5 bit)
) (
    clk, rst_n, sof, en, din, th4, feat_we, feat_blk, feat,
    frd_addr, frd_ch, frd_data
);

    // ---- derived widths (keep the module parametric in NCH / NF) ----
    function integer clog2f;
        input integer v;
        integer i;
        begin
            clog2f = 0;
            for (i = v - 1; i > 0; i = i >> 1) clog2f = clog2f + 1;
        end
    endfunction

    localparam integer NW = NF / 8;             // packed words per channel (32)
    localparam integer AW = clog2f(NW);         // frd_addr width (5)
    localparam integer CW = clog2f(NCH);        // frd_ch   width (3)
    localparam integer BW = clog2f(NF);         // block index width (8)
    localparam integer WW = NCH * 8;            // packed word width (64)

    input  wire                 clk;
    input  wire                 rst_n;      // async active-low
    input  wire                 sof;        // layer start: clear counters
    input  wire                 en;         // one pixel per cycle (block-major order)
    input  wire [NCH-1:0]       din;        // pixel activation bits
    input  wire [NCH*W_TH-1:0]  th4;        // channel c threshold in th4[c*W_TH +: W_TH]
    output reg                  feat_we;    // 1-cycle pulse at the end of each block
    output reg  [BW-1:0]        feat_blk;   // block index 0..NF-1 of that pulse
    output reg  [NCH-1:0]       feat;       // majority bits of that block
    // ---- feature read port for L5 (async distributed RAM) ----
    input  wire [AW-1:0]        frd_addr;   // word 0..NW-1
    input  wire [CW-1:0]        frd_ch;     // channel 0..NCH-1
    output wire [7:0]           frd_data;   // 8 blocks of that channel

    reg [4:0]     pcnt;                     // pixel index inside the block 0..15
    reg [BW-1:0]  blk;                      // block index 0..NF-1
    reg [2:0]     bpos;                     // block position inside the word 0..7
    reg [WW-1:0]  wacc;                     // packing shift register
    reg [WW-1:0]  bmem [0:NW-1];            // NW x WW distributed RAM (2048 bit @ 8ch)
    reg [4:0]     sum [0:NCH-1];            // running per-channel count 0..16

    // majority decision including the pixel that arrives this cycle
    reg [NCH-1:0] fn;
    integer c;
    always @* begin
        for (c = 0; c < NCH; c = c + 1)
            fn[c] = ((sum[c] + {4'b0, din[c]}) >= th4[c*W_TH +: W_TH]);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pcnt    <= 5'd0;
            blk     <= {BW{1'b0}};
            bpos    <= 3'd0;
            wacc    <= {WW{1'b0}};
            feat    <= {NCH{1'b0}};
            feat_blk<= {BW{1'b0}};
            feat_we <= 1'b0;
            for (c = 0; c < NCH; c = c + 1) sum[c] <= 5'd0;
        end else begin
            feat_we <= 1'b0;
            if (sof) begin
                pcnt    <= 5'd0;
                blk     <= {BW{1'b0}};
                bpos    <= 3'd0;
                wacc    <= {WW{1'b0}};
                feat    <= {NCH{1'b0}};
                feat_blk<= {BW{1'b0}};
                for (c = 0; c < NCH; c = c + 1) sum[c] <= 5'd0;
            end else if (en) begin
                if (pcnt == 5'd15) begin
                    // ---- block finished ----
                    pcnt    <= 5'd0;
                    for (c = 0; c < NCH; c = c + 1) sum[c] <= 5'd0;
                    feat    <= fn;
                    feat_we <= 1'b1;
                    feat_blk<= blk;
                    blk     <= blk + {{(BW-1){1'b0}}, 1'b1};
                    // pack right-to-left: block 8w+0 ends up in bits [NCH-1:0]
                    if (bpos == 3'd7) begin
                        bmem[blk[BW-1:3]] <= {fn, wacc[WW-1:NCH]};
                        wacc <= {WW{1'b0}};
                        bpos <= 3'd0;
                    end else begin
                        wacc <= {fn, wacc[WW-1:NCH]};
                        bpos <= bpos + 3'd1;
                    end
                end else begin
                    pcnt <= pcnt + 5'd1;
                    for (c = 0; c < NCH; c = c + 1) sum[c] <= sum[c] + {4'b0, din[c]};
                end
            end
        end
    end

    // ---- async read of the packed feature word, one channel selected ----
    genvar j;
    generate
        for (j = 0; j < 8; j = j + 1) begin : g_frd
            assign frd_data[j] = bmem[frd_addr][j*NCH + frd_ch];
        end
    endgenerate

endmodule
