// act_bram.v -- ACT_A / ACT_B activation ping-pong buffer (one read port + one write port)
//
// Each bank stores P planes x DEPTH pixels of 1 bit.  With the defaults
// (P=8, DEPTH=4096) one bank = 32 Kbit = one RAMB36, so the whole block is
// 2 x RAMB36, exactly the ACT_A/ACT_B budget of spec v2.2 6.4.
//
//   word  : one pixel,  bit p = plane p (bit=1 -> +1, spec 05 sec.1)
//   addr  : row*64 + col (0..4095, row-major, identical for every plane)
//
// Ports
//   * exactly one write port (we/wbank/waddr/wdata)
//   * exactly one read port  (re/rbank/raddr/rdata) with a REGISTERED output,
//     i.e. 1 cycle latency.  The consumer must present the address one cycle
//     before the data is needed; layer_seq does this with a prefetch cycle
//     (S_L2 cycle 0 / S_L3 cycle 0 / S_L4 cycle 0 present address 0).
//   * the bank select is per port, so L2 can read bank A and write bank B in
//     the same cycle.
//
// L3 writes only 4 planes.  The datapath drives wdata = {4'b0, act4}, so bank A
// planes 7:4 are zeroed at every address; L4 only reads planes 3:0, so this is
// harmless and no byte-enable port is needed.
//
// No reset: block RAM has no reset port.  Every address that is read has been
// written by the preceding layer (L1->A, L2->B, L3->A[3:0]), so no X propagates
// in normal operation.  This is the same convention as the WBRAM/FRAME BRAMs.
// Verilog-2001, synthesizable, no DSP.
`timescale 1ns / 1ps

module act_bram #(
    parameter integer P     = 8,      // planes per pixel
    parameter integer AW    = 12,     // address width
    parameter integer DEPTH = 4096    // pixels per bank
) (
    input  wire          clk,
    input  wire          we,          // write enable
    input  wire          wbank,       // 0 = bank A, 1 = bank B
    input  wire [AW-1:0] waddr,       // pixel address
    input  wire [P-1:0]  wdata,       // P plane bits
    input  wire          re,          // read enable
    input  wire          rbank,       // 0 = bank A, 1 = bank B
    input  wire [AW-1:0] raddr,       // pixel address
    output reg  [P-1:0]  rdata        // read data, 1 cycle after raddr
);

    // ---- unified store: bank bit folded into the address ----
    // (2026-11-01 resource fix, variant B) Two separate arrays plus a per-port bank
    // select confused block-RAM inference in Vivado 2022.2 and the store degraded to
    // 1,536 LUTRAM. A single array with {bank, addr} addresses is the canonical form.
    (* ram_style = "block" *) reg [P-1:0] mem [0:2*DEPTH-1];

    // ---- write port (one bank per cycle) ----
    always @(posedge clk) begin
        if (we) mem[{wbank, waddr}] <= wdata;
    end

    // ---- read port (one bank per cycle, registered output) ----
    // NOTE (patch2 resource fix, 2026-11-01): the ternary form
    //     rdata <= rbank ? mem_b[raddr] : mem_a[raddr];
    // prevented Vivado from inferring block RAM (the two arrays degraded to
    // 1,536 LUTRAM, pushing the whole design over the <=5,000 LUT gate).
    // Splitting into an if/else keeps the exact same 1-cycle registered read
    // behaviour but lets the arrays map to 2 x RAMB36.
    always @(posedge clk) begin
        if (re) rdata <= mem[{rbank, raddr}];
    end

endmodule
