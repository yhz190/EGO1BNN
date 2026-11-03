// sfr_bnn.v -- BNN SFR register file, base 0x0001_0000 (v2.2 sec.3.4)
//
//   off   name        R/W  bits
//   0x00  CTRL        W    bit0 START (launch one frame, 1-cycle pulse into the engine)
//                          bit1 CLR   (clear done / err / crc_err / class / result)
//                          bit2 GALLERY (accepted, not implemented in this baseline)
//   0x04  STATUS      R    bit0 busy, bit1 done (sticky), bit2 err (sticky, any error),
//                          bit3 frame_ready, bit4 crc_err (sticky), bit5 no_frame (sticky)
//   0x08  THRESH      W    gray threshold 0..255 (stored, used by the PC side)
//   0x0C  CLASS       R/W  firmware writes the final class -> also triggers the 10 B
//                          result frame; the value is readable again
//   0x10  SCORE0      R    19 bit signed Q6 score of class 0, sign extended to 32 bit
//   0x14  SCORE1      R    class 1
//   0x18  SCORE2      R    class 2
//   0x1C  FPS         R    frames per second (1 s window, FPS_WIN clk cycles)
//   0x20  SEQ         R    input frame sequence number (from frame_rx, dropped-frame check)
//   0x24  MODEL_ID    R    model / layout id (0x0001, blob bits [8616,8632))
//   0x28  MARGIN      R/W  top1-top2 reject threshold, Q6 scaled: 24 * 64 = 1536
//   0x2C  ERRORS      R    error counter (PCPI rejections + CRC failures + starts w/o frame)
//   0x30  SCORE3      R    class 3 (junk) score, not carried in the 10 B frame
//
// NOTE (v2.2 sec.3.4 vs this baseline): the frozen table has bit0 busy / bit1 done /
// bit2 err / bit3 frame_ready / bit4 crc_err / bit5 no_frame.  That is implemented
// verbatim here; the D4 task text writes "bit2 crc_err", which is covered by bit2 (err)
// being sticky on CRC failures AND by the dedicated bit4.  See the integration record.
//
// The SFR is a plain register file: no address decode, no memory here.  top_bnn owns
// the bus decode and guarantees that an SFR write never reaches IMEM/DMEM.
// Verilog-2001, synthesizable.
`timescale 1ns / 1ps

module sfr_bnn #(
    parameter [15:0]  MODEL_ID   = 16'h0001,
    parameter [15:0]  MARGIN_DEF = 16'd1536,      // Q6: 24 * 64
    parameter integer FPS_WIN    = 50000000       // 1 s @ 50 MHz
) (
    input  wire        clk,
    input  wire        rst_n,
    // ---- CPU bus (already decoded by top_bnn) ----
    input  wire        sel,
    input  wire        we,             // 1-cycle write strobe (mem_valid & mem_ready & wstrb!=0)
    input  wire [5:0]  addr,           // byte offset inside the 64 B window
    input  wire [31:0] wdata,
    output reg  [31:0] rdata,
    // ---- engine ----
    output reg         ctrl_start,     // 1-cycle pulse -> bnn_engine.start (CTRL.START)
    input  wire        eng_start,      // 1-cycle pulse: engine actually started (any path)
    input  wire        eng_busy,
    input  wire        eng_done,       // 1-cycle pulse (bnn_engine.all_done)
    input  wire        eng_ready,
    input  wire [75:0] score_i,
    input  wire [2:0]  cls_i,
    // ---- frame path ----
    input  wire        frame_new,      // 1-cycle: a good 517 B frame was received
    input  wire        crc_bad,        // 1-cycle: CRC failure
    input  wire [15:0] frame_seq,
    // ---- error strobe from the PCPI decoder ----
    input  wire        pcpi_err,
    // ---- result frame trigger / payload ----
    output reg         tx_go,          // 1-cycle: start the 10 B result frame
    output reg  [7:0]  class_o,
    output reg  [15:0] margin_o,
    output wire [15:0] errors_o,
    output wire [7:0]  fps_o,
    output wire [7:0]  frames_o,       // free running frame counter (result frame "fps" byte)
    output wire [7:0]  flags_o,
    output wire [7:0]  thresh_o,
    output wire [31:0] status_o        // STATUS word, address independent
);

    // ================================================================ registers
    reg        st_done, st_err, st_crc, st_nofrm, st_frame_rdy;
    reg [15:0] err_cnt;
    reg [7:0]  thresh_r;
    reg [7:0]  fps_cnt, fps_r;
    reg [31:0] fps_win;
    reg [15:0] frame_cnt;

    wire [18:0] sc0 = score_i[0*19 +: 19];
    wire [18:0] sc1 = score_i[1*19 +: 19];
    wire [18:0] sc2 = score_i[2*19 +: 19];
    wire [18:0] sc3 = score_i[3*19 +: 19];

    assign errors_o = err_cnt;
    assign fps_o    = fps_r;
    assign frames_o = frame_cnt[7:0];
    assign thresh_o = thresh_r;
    // STATUS is also exported as a standalone word: bnn.wait returns it while the CPU
    // is stalled on the instruction (the bus address at that moment is meaningless).
    assign status_o = {26'd0, st_nofrm, st_crc, st_frame_rdy, st_err, st_done, eng_busy};
    // result-frame flags: bit0 done / bit1 busy / bit2 crc_err (v2.2 sec.7.1)
    assign flags_o  = {5'd0, st_crc, eng_busy, st_done};

    // ================================================================ writes
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_start    <= 1'b0;
            tx_go         <= 1'b0;
            class_o       <= 8'd0;
            margin_o      <= MARGIN_DEF;
            thresh_r      <= 8'd0;
            st_done       <= 1'b0;
            st_err        <= 1'b0;
            st_crc        <= 1'b0;
            st_nofrm      <= 1'b0;
            st_frame_rdy  <= 1'b0;
            err_cnt       <= 16'd0;
            frame_cnt     <= 16'd0;
            fps_cnt       <= 8'd0;
            fps_r         <= 8'd0;
            fps_win       <= 32'd0;
        end else begin
            ctrl_start <= 1'b0;
            tx_go      <= 1'b0;

            // ---------------- CPU writes ----------------
            if (sel && we) begin
                case (addr)
                    6'h00: begin                                  // CTRL
                        // bit0 START -> 1-cycle pulse into the engine (alias of
                        // bnn.conv; v2.2 sec.5.3 keeps bnn.conv as the canonical path)
                        if (wdata[0]) ctrl_start <= 1'b1;
                        // bit2 GALLERY is accepted and ignored in this baseline
                        if (wdata[1]) begin                       // CLR
                            st_done  <= 1'b0;
                            st_err   <= 1'b0;
                            st_crc   <= 1'b0;
                            st_nofrm <= 1'b0;
                            class_o  <= 8'd0;
                        end
                    end
                    6'h08: thresh_r <= wdata[7:0];                // THRESH
                    6'h0C: begin                                  // CLASS
                        class_o <= wdata[7:0];
                        tx_go   <= 1'b1;                          // send the 10 B frame
                    end
                    6'h28: margin_o <= wdata[15:0];               // MARGIN
                    default: ;
                endcase
            end

            // ---------------- engine / frame events ----------------
            // any engine start (bnn.conv or CTRL.START) consumes the pending frame
            if (eng_start) begin
                st_done      <= 1'b0;
                st_frame_rdy <= 1'b0;
                if (!st_frame_rdy) st_nofrm <= 1'b1;   // inference without a new frame
            end
            if (eng_done) begin
                st_done   <= 1'b1;
                frame_cnt <= frame_cnt + 16'd1;
                fps_cnt   <= fps_cnt + 8'd1;
            end
            if (frame_new) begin
                st_frame_rdy <= 1'b1;
                st_nofrm     <= 1'b0;
            end
            if (crc_bad) begin
                st_crc <= 1'b1;
                st_err <= 1'b1;
                if (err_cnt != 16'hFFFF) err_cnt <= err_cnt + 16'd1;
            end
            if (pcpi_err) begin
                st_err <= 1'b1;
                if (err_cnt != 16'hFFFF) err_cnt <= err_cnt + 16'd1;
            end

            // ---------------- 1 s fps window ----------------
            if (fps_win == (FPS_WIN - 1)) begin
                fps_win <= 32'd0;
                fps_r   <= fps_cnt;
                fps_cnt <= 8'd0;
            end else begin
                fps_win <= fps_win + 32'd1;
            end
        end
    end

    // ================================================================ read mux
    always @* begin
        rdata = 32'hDEAD_0000;
        case (addr)
            6'h00: rdata = 32'd0;
            6'h04: rdata = status_o;
            6'h08: rdata = {24'd0, thresh_r};
            6'h0C: rdata = {24'd0, class_o};
            6'h10: rdata = {{13{sc0[18]}}, sc0};
            6'h14: rdata = {{13{sc1[18]}}, sc1};
            6'h18: rdata = {{13{sc2[18]}}, sc2};
            6'h1C: rdata = {24'd0, fps_r};
            6'h20: rdata = {16'd0, frame_seq};
            6'h24: rdata = {16'd0, MODEL_ID};
            6'h28: rdata = {16'd0, margin_o};
            6'h2C: rdata = {16'd0, err_cnt};
            6'h30: rdata = {{13{sc3[18]}}, sc3};
            default: rdata = 32'hDEAD_0000;
        endcase
    end

    wire unused_ok = &{1'b0, cls_i[0], frame_cnt[8]};

endmodule
