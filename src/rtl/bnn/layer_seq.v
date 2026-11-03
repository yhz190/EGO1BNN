// layer_seq.v -- D4 layer sequencer / scan controller (L1 -> L2 -> L3 -> L4 -> L5)
//
// FSM (spec v2.2 6.3 + patch1):
//
//   S_IDLE -> S_L1 -> S_L1_DONE -> S_L2 -> S_L3 -> S_L4 -> S_L5 -> S_DONE -> S_IDLE
//
//   S_L1   L1: FRAME -> ACT_A, 65x65 scan                  4225 shift beats
//   S_L1_DONE  explicit barrier (1 cycle): the last ACT_A write must be
//              visible before L2 issues its first ACT_A read
//   S_L2   L2: ACT_A -> ACT_B, 65x65 scan                  4225 shift beats
//   S_L3   L3: ACT_B -> ACT_A[7:0], 1x1 conv, no padding   4096 pixel beats
//   S_L4   L4: ACT_A[7:0] -> feature RAM (block-major scan) 4096 pixel beats
//   S_L5   L5: 2048 -> 4 FC                                256 beats (8 feats/beat)
//   S_DONE 1 cycle, all_done
//
// patch2 (v2.2-patch2): L3 has 8 output channels and L4 8 channels, but both
// stay fully parallel, so their beat/state counts are unchanged (4098 each).
// Only L5 grows: NF = 2048 -> 256 beats, i.e. S_L5 = 257 state cycles
// (256 beats + 1 output register).  Whole frame = 4228 + 1 + 4228 + 4098 + 4098
// + 257 + 1 = 16,911 cycles (sim / report/design_notes/25_G1拍数定界_L5配置与总拍数.md).
//
// Beat accounting of one L1/L2 layer (= 65 x 65 = 4225 shifts):
//   * 4096 data cycles   (din_we = 1, one ACT/FRAME pixel requested)
//   * 129 warm-up cycles (65 row-0 + 64 col-0), result discarded
//   * 4096 valid windows (win_valid from win_buf), written one cycle later
//   The state itself lasts 4228 cycles: cycle 0 is `sof` (win_buf reset),
//   cycles 1..4225 are the 4225 shifts, cycle 4226 is the last valid window and
//   cycle 4227 is the write of that last window (win_buf/accum output register).
//   layer_beats latches the *beat* count (4225), not the state length.
//
// Read prefetch: act_bram has a 1-cycle registered read, so:
//   * S_L2 cycle 0 presents address 0 (pixel 0 is needed at cycle 1)
//   * S_L3 cycle 0 presents address 0 (beat 0 is consumed at cycle 1)
//   * S_L4 cycle 0 presents the block-major address of beat 0
//   * inside the state, rd_addr is always the address of the pixel that will be
//     consumed on the NEXT cycle (L2: pixcnt + din_we, L3: cyc, L4: see below)
//   S_L3/S_L4 therefore run 4098 cycles for 4096 beats (1 head + 1 tail).
//
// L4 scan order is block-major: 16 consecutive pixels of one 4x4 block, then
// the next block (B = by*16+bx).  With 64 = 4*16 the address is a pure wiring
// concatenation {by[3:0], r[1:0], bx[3:0], c[1:0]} -> no adder, and l4_pool
// only needs one 4 bit "pixel inside block" counter instead of 16 parallel
// per-block counters (saves ~250 LUT).
//
// Write pipeline: win_valid / pix_en at cycle t -> wr_en at t+1 with the address
// captured at t, because accum / l3_mix have registered outputs.
//
// Verilog-2001, synthesizable, no DSP.
`timescale 1ns / 1ps

module layer_seq #(
    parameter integer N_SCAN = 4225,   // L1/L2 shift beats (65 x 65)
    parameter integer N_PIX  = 4096,   // L3/L4 pixel beats (64 x 64)
    parameter integer N_FC   = 256     // L5 beats (2048 features / 8)
) (
    input  wire        clk,
    input  wire        rst_n,       // async active-low
    input  wire        start,       // 1-cycle pulse: run one frame (ignored while busy)
    input  wire        win_valid,   // from win_buf: a 3x3 window is valid
    // ---- status ----
    output wire        busy,
    output reg  [2:0]  layer,       // current layer 1..5 (0 in S_IDLE / S_DONE)
    output reg         layer_done,  // 1-cycle pulse after each layer
    output reg  [2:0]  done_layer,  // layer number that just finished
    output wire        all_done,    // 1-cycle pulse in S_DONE (frame finished)
    output wire [2:0]  state,       // FSM state (TB / debug)
    output reg         layer_start, // 1-cycle pulse in the first cycle of each layer
    // ---- L1/L2 scan ----
    output wire        sof,         // win_buf start-of-layer (1 cycle)
    output wire        din_we,      // win_buf pixel valid
    output wire [11:0] pix_idx,     // data pixel index (FRAME address for L1)
    // ---- ACT read port (address for the NEXT cycle -> act_bram 1 cycle latency) ----
    output wire        rd_en,
    output wire        rd_bank,     // 0 = A, 1 = B
    output wire [11:0] rd_addr,
    // ---- ACT write port (registered result address) ----
    output wire        wr_en,
    output wire        wr_bank,     // 0 = A, 1 = B
    output wire        wr_lo4,      // 1 -> this write is the L3 result (legacy name;
                                    //      patch2 L3 writes all 8 planes, the flag is
                                    //      still needed to tell L3 from L1 on bank A)
    output wire [11:0] waddr,
    // ---- L3/L4 pixel beat + counters ----
    output wire        pix_en,      // 1 -> act read data is a valid pixel this cycle
    output reg  [12:0] beat_cnt,    // beats issued in the current layer
    output reg  [12:0] win_cnt,     // valid outputs (L1/L2 windows, L3/L4 pixels, L5 beats)
    output wire [12:0] cyc_cnt,     // cycle index inside the current state
    output reg  [12:0] layer_beats  // beat count of the layer that just finished
);

    localparam [2:0] S_IDLE = 3'd0, S_L1 = 3'd1, S_L1_DONE = 3'd2, S_L2 = 3'd3,
                     S_L3 = 3'd4, S_L4 = 3'd5, S_L5 = 3'd6, S_DONE = 3'd7;

    localparam [12:0] LAST_SCAN = N_SCAN + 2;   // 4227: last write of the 65x65 scan
    localparam [12:0] LAST_PIX  = N_PIX  + 1;   // 4097: last pixel write
    localparam [12:0] LAST_FC   = N_FC;         // 256 : score valid (257th state cycle)

    reg [2:0]  st;
    reg [12:0] cyc;
    reg [6:0]  row, col;             // 65x65 stream position
    reg [11:0] pixcnt;               // next data pixel index (L1/L2)
    reg [11:0] pix_beat;             // L3 pixel beat index
    reg [11:0] wcnt;                 // L1/L2 window index
    reg        wr_q;
    reg [11:0] wad_q;
    reg        wrb_q;
    reg        lo4_q;

    assign state    = st;
    assign busy     = (st != S_IDLE);
    assign all_done = (st == S_DONE);
    assign cyc_cnt  = cyc;

    // ---------------------------------------------------------------- scan
    wire in_l12 = (st == S_L1) || (st == S_L2);

    assign sof    = in_l12 && (cyc == 13'd0);
    assign din_we = in_l12 && (cyc >= 13'd1) && (cyc <= N_SCAN) &&
                    (row[6] == 1'b0) && (col[6] == 1'b0);     // row<64 && col<64
    assign pix_idx = pixcnt;

    // ---------------------------------------------------------------- read port
    assign pix_en  = ((st == S_L3) || (st == S_L4)) && (cyc >= 13'd1) && (cyc <= N_PIX);
    assign rd_en   = (st == S_L2) || (st == S_L3) || (st == S_L4);
    assign rd_bank = (st == S_L3);                                // L3 reads B

    // next data pixel of the 65x65 stream (address of the pixel needed at t+1)
    wire [12:0] rd_nx13 = {1'b0, pixcnt} + {12'd0, din_we};
    wire [11:0] rd_l2   = (rd_nx13 > 13'd4095) ? 12'd4095 : rd_nx13[11:0];
    wire [11:0] rd_l3   = (cyc <= (N_PIX - 1)) ? cyc[11:0] : 12'd4095;

    // L4 block-major address: {by, r, bx, c} with B = by*16+bx, pixel = r*64+c
    wire [7:0]  l4_blk  = cyc[11:4];
    wire [11:0] l4_addr = {l4_blk[7:4], cyc[3:2], l4_blk[3:0], cyc[1:0]};

    assign rd_addr = (st == S_L2) ? rd_l2 :
                     (st == S_L3) ? rd_l3 : l4_addr;

    // ---------------------------------------------------------------- write port
    assign wr_en   = wr_q;
    assign waddr   = wad_q;
    assign wr_bank = wrb_q;
    assign wr_lo4  = lo4_q;

    // ---------------------------------------------------------------- FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st          <= S_IDLE;
            cyc         <= 13'd0;
            layer       <= 3'd0;
            layer_done  <= 1'b0;
            done_layer  <= 3'd0;
            layer_start <= 1'b0;
            layer_beats <= 13'd0;
            beat_cnt    <= 13'd0;
            win_cnt     <= 13'd0;
            row         <= 7'd0;
            col         <= 7'd0;
            pixcnt      <= 12'd0;
            pix_beat    <= 12'd0;
            wcnt        <= 12'd0;
            wr_q        <= 1'b0;
            wad_q       <= 12'd0;
            wrb_q       <= 1'b0;
            lo4_q       <= 1'b0;
        end else begin
            layer_done  <= 1'b0;
            layer_start <= 1'b0;
            wr_q        <= 1'b0;

            // ---- one-cycle write pipeline behind win_valid / pix_en ----
            if (st == S_L1 && win_valid) begin
                wr_q <= 1'b1; wad_q <= wcnt; wrb_q <= 1'b0; lo4_q <= 1'b0;
                wcnt <= wcnt + 12'd1;
            end else if (st == S_L2 && win_valid) begin
                wr_q <= 1'b1; wad_q <= wcnt; wrb_q <= 1'b1; lo4_q <= 1'b0;
                wcnt <= wcnt + 12'd1;
            end else if (st == S_L3 && pix_en) begin
                wr_q <= 1'b1; wad_q <= pix_beat; wrb_q <= 1'b0; lo4_q <= 1'b1;
            end

            case (st)
            // ---------------------------------------------------- idle
            S_IDLE: begin
                cyc   <= 13'd0;
                layer <= 3'd0;
                if (start) begin
                    st          <= S_L1;
                    layer       <= 3'd1;
                    layer_start <= 1'b1;
                    cyc         <= 13'd0;
                    row         <= 7'd0;
                    col         <= 7'd0;
                    pixcnt      <= 12'd0;
                    pix_beat    <= 12'd0;
                    wcnt        <= 12'd0;
                    beat_cnt    <= 13'd0;
                    win_cnt     <= 13'd0;
                end
            end
            // ---------------------------------------------------- L1 / L2 scan
            S_L1, S_L2: begin
                if (cyc == LAST_SCAN) begin
                    cyc         <= 13'd0;
                    layer_done  <= 1'b1;
                    done_layer  <= (st == S_L1) ? 3'd1 : 3'd2;
                    layer_beats <= beat_cnt;
                    if (st == S_L1) begin
                        st    <= S_L1_DONE;
                        layer <= 3'd1;                    // stay in L1 during the barrier
                    end else begin
                        st          <= S_L3;
                        layer       <= 3'd3;
                        layer_start <= 1'b1;
                        row         <= 7'd0;
                        col         <= 7'd0;
                        pixcnt      <= 12'd0;
                        pix_beat    <= 12'd0;
                        wcnt        <= 12'd0;
                        beat_cnt    <= 13'd0;
                        win_cnt     <= 13'd0;
                    end
                end else begin
                    cyc <= cyc + 13'd1;
                    if ((cyc >= 13'd1) && (cyc <= N_SCAN)) begin
                        beat_cnt <= beat_cnt + 13'd1;
                        if ((row[6] == 1'b0) && (col[6] == 1'b0))
                            pixcnt <= pixcnt + 12'd1;
                        if (col == 7'd64) begin
                            col <= 7'd0;
                            row <= row + 7'd1;
                        end else begin
                            col <= col + 7'd1;
                        end
                    end
                    if (win_valid) win_cnt <= win_cnt + 13'd1;
                end
            end
            // ---------------------------------------------------- L1 -> L2 barrier
            S_L1_DONE: begin
                st          <= S_L2;
                layer       <= 3'd2;
                layer_start <= 1'b1;
                cyc         <= 13'd0;
                row         <= 7'd0;
                col         <= 7'd0;
                pixcnt      <= 12'd0;
                pix_beat    <= 12'd0;
                wcnt        <= 12'd0;
                beat_cnt    <= 13'd0;
                win_cnt     <= 13'd0;
            end
            // ---------------------------------------------------- L3
            S_L3: begin
                if (cyc == LAST_PIX) begin
                    cyc         <= 13'd0;
                    layer_done  <= 1'b1;
                    done_layer  <= 3'd3;
                    layer_beats <= beat_cnt;
                    st          <= S_L4;
                    layer       <= 3'd4;
                    layer_start <= 1'b1;
                    pix_beat    <= 12'd0;
                    beat_cnt    <= 13'd0;
                    win_cnt     <= 13'd0;
                end else begin
                    cyc <= cyc + 13'd1;
                    if (pix_en) begin
                        beat_cnt <= beat_cnt + 13'd1;
                        win_cnt  <= win_cnt  + 13'd1;
                        pix_beat <= pix_beat + 12'd1;
                    end
                end
            end
            // ---------------------------------------------------- L4
            S_L4: begin
                if (cyc == LAST_PIX) begin
                    cyc         <= 13'd0;
                    layer_done  <= 1'b1;
                    done_layer  <= 3'd4;
                    layer_beats <= beat_cnt;
                    st          <= S_L5;
                    layer       <= 3'd5;
                    layer_start <= 1'b1;
                    beat_cnt    <= 13'd0;
                    win_cnt     <= 13'd0;
                end else begin
                    cyc <= cyc + 13'd1;
                    if (pix_en) begin
                        beat_cnt <= beat_cnt + 13'd1;
                        win_cnt  <= win_cnt  + 13'd1;
                    end
                end
            end
            // ---------------------------------------------------- L5
            S_L5: begin
                if (cyc == LAST_FC) begin
                    cyc         <= 13'd0;
                    layer_done  <= 1'b1;
                    done_layer  <= 3'd5;
                    layer_beats <= beat_cnt;
                    st          <= S_DONE;
                    layer       <= 3'd0;
                end else begin
                    cyc <= cyc + 13'd1;
                    if (cyc < N_FC) begin
                        beat_cnt <= beat_cnt + 13'd1;
                        win_cnt  <= win_cnt  + 13'd1;
                    end
                end
            end
            // ---------------------------------------------------- done
            S_DONE: begin
                st    <= S_IDLE;
                cyc   <= 13'd0;
                layer <= 3'd0;
            end
            default: begin
                st    <= S_IDLE;
                cyc   <= 13'd0;
                layer <= 3'd0;
            end
            endcase
        end
    end

endmodule
