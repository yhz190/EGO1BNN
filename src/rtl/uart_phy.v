// uart_phy.v -- UART physical layer (rx + tx), 8N1, parameterisable divider
// Measured on EGO1: N5 = FPGA RX, T4 = FPGA TX, 115200 works with DIV=868 @100MHz
`timescale 1ns / 1ps

module uart_rx #(
    parameter integer DIV = 868          // 100e6 / 115200
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid
);
    // 2-FF synchroniser
    reg [1:0] sync;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) sync <= 2'b11;
        else        sync <= {sync[0], rx};
    wire line = sync[1];

    reg        busy;
    reg [11:0] cnt;
    reg [3:0]  bit_i;
    reg [7:0]  sh;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; cnt <= 12'd0; bit_i <= 4'd0; valid <= 1'b0; data <= 8'h00; sh <= 8'h00;
        end else begin
            valid <= 1'b0;
            if (!busy) begin
                if (!line) begin                       // start-bit edge
                    busy  <= 1'b1;
                    cnt   <= DIV + DIV / 2;            // sample at 1.5 bit = middle of data0
                    bit_i <= 4'd0;
                end
            end else begin
                if (cnt == 12'd0) begin
                    cnt <= DIV - 1;
                    if (bit_i < 4'd8) begin
                        sh[bit_i] <= line;             // LSB first
                        bit_i     <= bit_i + 4'd1;
                    end else begin
                        busy <= 1'b0;                  // stop bit
                        if (line) begin
                            data  <= sh;
                            valid <= 1'b1;
                        end
                    end
                end else begin
                    cnt <= cnt - 12'd1;
                end
            end
        end
    end
endmodule


module uart_tx #(
    parameter integer DIV = 868
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx,
    output reg        busy,
    output reg        done
);
    reg [11:0] cnt;
    reg [3:0]  bit_i;
    reg [9:0]  sh;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx <= 1'b1; busy <= 1'b0; done <= 1'b0; cnt <= 12'd0; bit_i <= 4'd0; sh <= 10'h3FF;
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start) begin
                    sh    <= {1'b1, data, 1'b0};       // stop, data[7:0], start
                    busy  <= 1'b1;
                    bit_i <= 4'd0;
                    cnt   <= DIV - 1;
                    tx    <= 1'b0;
                end
            end else begin
                if (cnt == 12'd0) begin
                    cnt <= DIV - 1;
                    if (bit_i < 4'd9) begin
                        bit_i <= bit_i + 4'd1;
                        tx    <= sh[bit_i + 4'd1];
                    end else begin
                        busy <= 1'b0;
                        tx   <= 1'b1;
                        done <= 1'b1;
                    end
                end else begin
                    cnt <= cnt - 12'd1;
                end
            end
        end
    end
endmodule
