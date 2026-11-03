// frame_rx_tx.v -- D2 protocol: parse 517B downlink frame, reply with 10B result frame
//
// Downlink (PC -> FPGA), 517 B:
//   A5 | 10 | len_lo | len_hi (=0x00,0x02) | payload[512] | CRC8
//   CRC8 covers ALL preceding bytes INCLUDING the 0xA5 header (decision 2026-10-31,
//   see report/design_notes/18_协议细节确认_CRC覆盖范围.md), poly 0x07, init 0x00.
// Uplink (FPGA -> PC), 10 B:
//   A5 | 83 | class | s0 | s1 | s2 | flags | fps | seq | CRC8(same coverage)
`timescale 1ns / 1ps

module frame_rx (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  b,
    input  wire        bvalid,
    output reg         fvalid,        // one-cycle pulse on a good frame
    output reg  [15:0] popcnt,        // popcount of the 512 payload bytes
    output reg  [15:0] seq,
    output reg  [15:0] n_ok,
    output reg  [15:0] n_err
);
    localparam [2:0] S_SOF = 3'd0, S_CMD = 3'd1, S_LENL = 3'd2,
                     S_LENH = 3'd3, S_PAY = 3'd4, S_CRC = 3'd5;

    reg [2:0]  st;
    reg [8:0]  cnt;
    reg [7:0]  crc;
    reg [12:0] pc;

    function [7:0] fmix;
        input [7:0] c;
        input [7:0] d;
        integer i;
        begin
            fmix = c ^ d;
            for (i = 0; i < 8; i = i + 1)
                fmix = fmix[7] ? ((fmix << 1) ^ 8'h07) : (fmix << 1);
        end
    endfunction

    function [3:0] pop8;
        input [7:0] v;
        begin
            pop8 = {3'b000, v[0]} + {3'b000, v[1]} + {3'b000, v[2]} + {3'b000, v[3]}
                 + {3'b000, v[4]} + {3'b000, v[5]} + {3'b000, v[6]} + {3'b000, v[7]};
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_SOF; cnt <= 9'd0; crc <= 8'h00; pc <= 13'd0;
            fvalid <= 1'b0; popcnt <= 16'd0; seq <= 16'd0; n_ok <= 16'd0; n_err <= 16'd0;
        end else begin
            fvalid <= 1'b0;
            if (bvalid) begin
                case (st)
                    S_SOF: begin
                        crc <= 8'h00;
                        if (b == 8'hA5) begin
                            crc <= fmix(8'h00, b);
                            st  <= S_CMD;
                        end
                    end
                    S_CMD: begin
                        if (b == 8'h10) begin crc <= fmix(crc, b); st <= S_LENL; end
                        else begin n_err <= n_err + 16'd1; st <= S_SOF; end
                    end
                    S_LENL: begin
                        if (b == 8'h00) begin crc <= fmix(crc, b); st <= S_LENH; end
                        else begin n_err <= n_err + 16'd1; st <= S_SOF; end
                    end
                    S_LENH: begin
                        if (b == 8'h02) begin
                            crc <= fmix(crc, b); cnt <= 9'd0; pc <= 13'd0; st <= S_PAY;
                        end else begin n_err <= n_err + 16'd1; st <= S_SOF; end
                    end
                    S_PAY: begin
                        crc <= fmix(crc, b);
                        pc  <= pc + {9'd0, pop8(b)};
                        if (cnt == 9'd511) st <= S_CRC;
                        else               cnt <= cnt + 9'd1;
                    end
                    S_CRC: begin
                        if (b == crc) begin
                            fvalid <= 1'b1;
                            popcnt <= {3'd0, pc};
                            seq    <= seq + 16'd1;
                            n_ok   <= n_ok + 16'd1;
                        end else begin
                            n_err <= n_err + 16'd1;
                        end
                        st <= S_SOF;
                    end
                    default: st <= S_SOF;
                endcase
            end
        end
    end
endmodule


module frame_tx (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [7:0]  cls,
    input  wire [7:0]  s0,
    input  wire [7:0]  s1,
    input  wire [7:0]  s2,
    input  wire [7:0]  flags,
    input  wire [7:0]  fps,
    input  wire [7:0]  seq8,
    output reg  [7:0]  tx_byte,
    output reg         tx_start,
    input  wire        tx_done,
    output reg         busy
);
    localparam [1:0] IDLE = 2'd0, ISSUE = 2'd1, WAIT = 2'd2;

    reg [1:0] st;
    reg [3:0] idx;
    reg [7:0] crc;

    function [7:0] fmix;
        input [7:0] c;
        input [7:0] d;
        integer i;
        begin
            fmix = c ^ d;
            for (i = 0; i < 8; i = i + 1)
                fmix = fmix[7] ? ((fmix << 1) ^ 8'h07) : (fmix << 1);
        end
    endfunction

    function [7:0] sel;
        input [3:0] i;
        input [7:0] c, a0, a1, a2, f, fp, sq, cr;
        begin
            case (i)
                4'd0: sel = 8'hA5;
                4'd1: sel = 8'h83;
                4'd2: sel = c;
                4'd3: sel = a0;
                4'd4: sel = a1;
                4'd5: sel = a2;
                4'd6: sel = f;
                4'd7: sel = fp;
                4'd8: sel = sq;
                default: sel = cr;
            endcase
        end
    endfunction

    wire [7:0] cur = sel(idx, cls, s0, s1, s2, flags, fps, seq8, crc);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= IDLE; idx <= 4'd0; crc <= 8'h00;
            tx_byte <= 8'h00; tx_start <= 1'b0; busy <= 1'b0;
        end else begin
            tx_start <= 1'b0;
            case (st)
                IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        idx  <= 4'd0;
                        crc  <= 8'h00;
                        busy <= 1'b1;
                        st   <= ISSUE;
                    end
                end
                ISSUE: begin
                    tx_byte  <= cur;
                    tx_start <= 1'b1;
                    st       <= WAIT;
                end
                WAIT: begin
                    if (tx_done) begin
                        if (idx == 4'd9) begin
                            st   <= IDLE;
                            busy <= 1'b0;
                        end else begin
                            crc <= fmix(crc, cur);      // only bytes 0..8 feed the CRC
                            idx <= idx + 4'd1;
                            st  <= ISSUE;
                        end
                    end
                end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
