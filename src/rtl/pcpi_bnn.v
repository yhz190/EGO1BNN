// pcpi_bnn.v -- PicoRV32 PCPI decoder for the v2.2 Custom-0 BNN instructions
//
// opcode = CUSTOM0 = 0x0B  (v2.2 sec.5.1)
//
//   funct3=0  bnn.conv   funct7 = start layer   -> pulse eng_start (1 cycle), pcpi_wr=0
//   funct3=1  bnn.wait   rd = status word (bit0 busy / bit1 done / bit2 err),
//                        pcpi_wait=1 while the engine runs, pcpi_ready + pcpi_wr when idle
//   funct3=2  bnn.acc    funct7 = class 0..3 -> rd = score[class], 19 bit signed
//                        sign extended to 32 bit
//
// funct7 for bnn.conv: the layer_seq FSM of this baseline can only enter at L1, so
// 0..5 all launch a FULL frame (L1..L5).  v2.2 sec.5.1 also asks for "1..4 = start
// from layer N"; that needs a layer_seq start-layer input (module is frozen) and is
// recorded as an open item.  funct7 > 5 -> error.
//
// HARD RULE (v2.2 sec.5.2 / R5): an unknown funct3, an out-of-range funct7, a conv
// while busy and a non-CUSTOM0 trap ALL assert pcpi_ready with pcpi_wr=0 in the same
// cycle.  PicoRV32 stalls forever in cpu_state_ld_rs1 waiting for pcpi_ready (with
// ENABLE_IRQ=0 there is no illegal-instruction trap), so "never leave ready low" is
// what keeps the CPU alive.
//
// Timing: PicoRV32 raises pcpi_valid one cycle after entering cpu_state_ld_rs1 and
// samples pcpi_ready combinationally in the SAME cycle, so all outputs here are
// combinational except the error pulse.  That gives the documented "1 cycle return"
// for conv/wait/acc.
//
// Verilog-2001, synthesizable.
`timescale 1ns / 1ps

module pcpi_bnn (
    input  wire        clk,
    input  wire        rst_n,
    // ---- PCPI ----
    input  wire        pcpi_valid,
    input  wire [31:0] pcpi_insn,
    input  wire [31:0] pcpi_rs1,
    input  wire [31:0] pcpi_rs2,
    output reg         pcpi_wr,
    output reg  [31:0] pcpi_rd,
    output reg         pcpi_wait,
    output reg         pcpi_ready,
    // ---- engine ----
    output reg         eng_start,      // 1-cycle pulse into bnn_engine.start
    input  wire        eng_busy,
    input  wire        eng_ready,      // weights loaded
    input  wire [75:0] score_i,
    // ---- status (same word as SFR STATUS) ----
    input  wire [31:0] status_i,
    // ---- error strobe (counted in SFR ERRORS) ----
    output reg         err_pulse
);

    localparam [6:0] OP_CUSTOM0 = 7'h0B;

    wire [6:0] opcode = pcpi_insn[6:0];
    wire [2:0] f3     = pcpi_insn[14:12];
    wire [6:0] f7     = pcpi_insn[31:25];
    wire       is_bnn = (opcode == OP_CUSTOM0);

    // a conv that has been accepted but is not yet visible as busy
    reg conv_pend;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          conv_pend <= 1'b0;
        else if (eng_start)  conv_pend <= 1'b1;
        else if (eng_busy)   conv_pend <= 1'b0;
    end
    wire busy_eff = eng_busy | conv_pend;

    // class-2 score slice (f7 is a variable index -> 4:1 mux of 19 bit)
    wire [1:0]  cls_sel = f7[1:0];
    wire [18:0] sc_sel   = score_i[cls_sel*19 +: 19];

    reg bad;

    always @* begin
        pcpi_wr    = 1'b0;
        pcpi_rd    = 32'd0;
        pcpi_wait  = 1'b0;
        pcpi_ready = 1'b0;
        eng_start  = 1'b0;
        bad        = 1'b0;

        if (pcpi_valid) begin
            if (!is_bnn) begin
                // NOT a custom-0 instruction: leave ready/wait/rd alone so that the
                // picorv32 MUL / DIV PCPI coprocessors (ENABLE_MUL etc.) keep working.
                // A truly unknown opcode then falls into picorv32's CATCH_ILLINSN path
                // (timeout/trap) instead of being silently swallowed here.
            end else begin
                case (f3)
                // ------------------------------------------------ bnn.conv
                3'd0: begin
                    pcpi_ready = 1'b1;                 // always ready, never blocking
                    if ((f7 <= 7'd5) && eng_ready && !busy_eff) eng_start = 1'b1;
                    else                                        bad       = 1'b1;
                end
                // ------------------------------------------------ bnn.wait
                3'd1: begin
                    if (busy_eff) begin
                        pcpi_wait = 1'b1;               // hold the CPU until done
                    end else begin
                        pcpi_ready = 1'b1;
                        pcpi_wr    = 1'b1;
                        pcpi_rd    = status_i;
                    end
                end
                // ------------------------------------------------ bnn.acc
                3'd2: begin
                    pcpi_ready = 1'b1;
                    if (f7 <= 7'd3) begin
                        pcpi_wr = 1'b1;
                        pcpi_rd = {{13{sc_sel[18]}}, sc_sel};   // sign extend 19 -> 32
                    end else begin
                        bad = 1'b1;
                    end
                end
                // ------------------------------------------------ unknown funct3
                default: begin
                    pcpi_ready = 1'b1;
                    bad        = 1'b1;
                end
                endcase
            end

            // error response: ready but no write-back (v2.2 sec.5.2)
            if (bad) begin
                pcpi_ready = 1'b1;
                pcpi_wr    = 1'b0;
                eng_start  = 1'b0;
            end
        end
    end

    // one-cycle error strobe -> SFR ERRORS counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) err_pulse <= 1'b0;
        else        err_pulse <= pcpi_valid && bad;
    end

    // pcpi_rs1/rs2 (frame base / weight base) are part of the ISA contract, but this
    // baseline addresses FRAME/WBRAM with fixed engine ports, so they are unused here.
    wire unused_ok = &{1'b0, pcpi_rs1[0], pcpi_rs2[0]};

endmodule
