// top_bnn.v -- EGO1 (XC7A35T-1CSG324C) BNN SoC top
//
//   sys_clk_100m (P17) --/2--> BUFG --> 50 MHz system clock (single clock domain)
//   sys_rst_n    (P15) --> 3 FF synchroniser --> rst_n (async assert, sync release)
//
//   PicoRV32 (ENABLE_PCPI=1, PROGADDR_RESET=0, STACKADDR=0x2FFC)
//     + IMEM  4 KB @ 0x0000_0000  (ROM, $readmemh fw.hex)
//     + DMEM  4 KB @ 0x0000_2000  (read/write, stack at 0x2FFC)
//     + WBRAM 2 KB @ 0x0002_0000  (271 words used, ROM, $readmemh weights.hex)
//     + SFR   64 B  @ 0x0001_0000  (sfr_bnn)
//   bnn_engine (layer_seq/win_buf/lane_array/accum/theta_cmp/act_bram/l3_mix/l4_pool/l5_fc)
//   pcpi_bnn   (Custom-0 bnn.conv / bnn.wait / bnn.acc)
//   frame_rx + payload capture -> engine FRAME (517 B downlink, 2-bank double buffer),
//   frame_tx + uart_tx (10 B result frame uplink)
//
// ============================== memory map / decode ==============================
//   addr[31:16] == 0x0000 : addr[13] = 0 -> IMEM  (0x0000..0x1FFF, 4 KB aliased twice)
//                                    1 -> DMEM  (0x2000..0x2FFF; 0x3000.. is unmapped)
//   addr[31:16] == 0x0001 : SFR   0x0001_0000 + addr[5:0]
//   addr[31:16] == 0x0002 : WBRAM 0x0002_0000 + addr[8:0]   (CPU read only)
//
//   D6: IMEM and DMEM are 1024 x 32 (4 KB) each, so the whole design fits the T6
//   budget (5 x RAMB36 + 2 x RAMB18 = 6.0 equivalents, GALLERY reserve included).
//   0x1000..0x1FFF aliases IMEM[0..1023] (the 10-bit word index drops addr[12]); the
//   firmware lives below 0x1000, so the alias is harmless.  DMEM does *not* alias
//   0x3000..0x3FFF: that range is decoded as unmapped and returns 0xDEAD_0000.
//
//   *** WRITE DECODE (the historic "SFR write leaking into IMEM" bug) ***
//   An SFR write can never reach IMEM/DMEM: imem_we requires addr[31:16]==0x0000,
//   the SFR requires 0x0001, and the three enables are mutually exclusive by
//   construction (one-hot on addr[31:16]/addr[13]).  Writes to WBRAM and IMEM from
//   the CPU are dropped on purpose (both are ROMs after $readmemh).
//
// ============================== CPU bus timing ==============================
//   picorv32 standard memory interface: mem_ready is registered, so every access is
//   2 cycles (address in cycle 1, registered read data + ready in cycle 2).  This
//   lets IMEM/DMEM/WBRAM/FRAME all be block RAM instead of a 2K:1 async LUT mux.
//
// ============================== boot order ==============================
//   rst_n release -> weight loader streams the 271 word v5 blob out of WBRAM into
//   bnn_engine (~271 + 36 cycles; the FC field is beat-major so it is written straight
//   into l5_fc) -> bnn_engine.ready -> cpu_rst_n released.
//   The CPU therefore can never issue bnn.conv before the engine has its weights.
//
// Verilog-2001, synthesizable, no DSP.
`timescale 1ns / 1ps

module top_bnn #(
    parameter integer BAUD_DIV = 54,          // 921600 baud @ 50 MHz (54 = 50e6/921600)
    parameter integer FPS_WIN  = 50000000,     // 1 s @ 50 MHz
    parameter integer WB_INIT  = 1,            // 1 -> $readmemh("weights.hex")
    parameter [15:0]  MODEL_ID = 16'h0001
) (
    input  wire        sys_clk_100m,       // P17
    input  wire        sys_rst_n,          // P15
    input  wire        pc_uart_rxd,        // N5
    output wire        pc_uart_txd,        // T4
    output wire [3:0]  led,                // {act, err, done, busy}
    // ---- VGA 640x480@60 (T5): driven by u_vga below, no framebuffer ----
    output wire        vga_hs,
    output wire        vga_vs,
    output wire [11:0] vga_rgb
);

    // ================================================================ clock / reset
    // clk_div / rst_sync are power-up-initialised (Xilinx FF INIT=0): the reset
    // synchroniser must start at 0 so the async resets really assert at power-on,
    // otherwise simulation (and a cold FPGA) would never leave the X state.
    reg clk_div;
    initial clk_div = 1'b0;
    always @(posedge sys_clk_100m) clk_div <= ~clk_div;

    wire clk;                                  // 50 MHz, BUFG buffered
    BUFG u_bufg (.I(clk_div), .O(clk));

    reg [2:0] rst_sync;
    initial rst_sync = 3'd0;
    always @(posedge clk) rst_sync <= {rst_sync[1:0], sys_rst_n};
    wire rst_n = rst_sync[2];

    // declared early: used by the SFR / PCPI / frame instances below
    wire frx_err_pulse;
    wire frx_fvalid;
    wire [31:0] sfr_status;

    // ================================================================ CPU
    wire        mem_valid, mem_ready, mem_instr;
    wire [31:0] mem_addr, mem_wdata, mem_rdata;
    wire [3:0]  mem_wstrb;
    wire        trap;

    wire        pcpi_valid, pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_insn, pcpi_rs1, pcpi_rs2, pcpi_rd;

    wire        eng_ready;
    reg  [2:0]  cpu_rst_sync;
    initial cpu_rst_sync = 3'd0;
    always @(posedge clk) cpu_rst_sync <= {cpu_rst_sync[1:0], (rst_n & eng_ready)};
    wire        cpu_rst_n = cpu_rst_sync[2];

    picorv32 #(
        .ENABLE_PCPI(1),
        .ENABLE_MUL(1),
        .ENABLE_FAST_MUL(0),
        .ENABLE_DIV(0),
        .ENABLE_IRQ(0),
        .COMPRESSED_ISA(0),
        .ENABLE_COUNTERS(1),
        .ENABLE_COUNTERS64(0),
        .REGS_INIT_ZERO(1),          // matches the Xilinx power-up state (INIT=0)
        .PROGADDR_RESET(32'h0000_0000),
        .STACKADDR(32'h0000_2FFC)
    ) u_cpu (
        .clk(clk), .resetn(cpu_rst_n), .trap(trap),
        .mem_valid(mem_valid), .mem_instr(mem_instr), .mem_ready(mem_ready),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata), .mem_wstrb(mem_wstrb),
        .mem_rdata(mem_rdata),
        .mem_la_read(), .mem_la_write(), .mem_la_addr(), .mem_la_wdata(),
        .mem_la_wstrb(),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd), .pcpi_wait(pcpi_wait),
        .pcpi_ready(pcpi_ready),
        .irq(32'd0), .eoi()
    );

    // ---- registered 2-cycle response ----
    reg mem_ready_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) mem_ready_r <= 1'b0;
        else        mem_ready_r <= mem_valid && !mem_ready_r;
    end
    assign mem_ready = mem_ready_r;

    // ================================================================ address decode
    wire sel_imem = (mem_addr[31:16] == 16'h0000) && (mem_addr[13] == 1'b0);
    wire sel_dmem = (mem_addr[31:16] == 16'h0000) && (mem_addr[13] == 1'b1) &&
                    (mem_addr[12] == 1'b0);                    // 4 KB: 0x2000..0x2FFF only
    wire sel_sfr  = (mem_addr[31:16] == 16'h0001);
    wire sel_wb   = (mem_addr[31:16] == 16'h0002);

    wire [9:0]  iidx = mem_addr[11:2];                         // 1024 words = 4 KB
    wire [8:0]  widx = mem_addr[8:0];

    // one write strobe, decoded one-hot (IMEM/DMEM/WBRAM are ROMs -> writes dropped)
    wire wr_req  = mem_valid && mem_ready_r && (mem_wstrb != 4'b0000);
    wire dmem_we = wr_req && sel_dmem;
    wire sfr_we  = wr_req && sel_sfr;
    wire imem_we = wr_req && sel_imem && !sel_sfr && !sel_wb;   // explicit guard
    wire wb_we   = wr_req && sel_wb;                            // dropped (ROM)

    // ================================================================ IMEM / DMEM
    reg [31:0] imem [0:1023];
    reg [31:0] dmem [0:1023];
    reg [31:0] wbmem [0:511];

    initial begin
        $readmemh("fw.hex", imem);
        if (WB_INIT != 0) $readmemh("weights.hex", wbmem);
    end

    reg [31:0] imem_q, dmem_q;
    always @(posedge clk) begin
        if (mem_valid && sel_imem) imem_q <= imem[iidx];
        if (mem_valid && sel_dmem) dmem_q <= dmem[iidx];
        if (dmem_we)               dmem[iidx] <= mem_wdata;
        if (imem_we)               imem[iidx] <= mem_wdata;
        if (wb_we)                 ; // WBRAM is read-only after $readmemh
    end

    // ================================================================ WBRAM (one read port)
    // The weight loader owns the port while it runs (the CPU is still in reset then),
    // the CPU uses it afterwards (CRC self-check).  No runtime write port.
    reg  [8:0]  wl_addr;
    reg         wl_run;
    wire [8:0]  wb_raddr = wl_run ? wl_addr : widx;
    reg  [31:0] wb_q;
    always @(posedge clk) wb_q <= wbmem[wb_raddr];

    // ================================================================ weight loader
    // streams 271 words (continuous) into bnn_engine, which unpacks weights + theta
    // and streams the beat-major L5 field straight into l5_fc; bnn_engine.ready rises
    // when it is done.
    reg  [3:0] wl_dly;
    reg        wl_req;                 // read request valid (data next cycle)
    reg        wl_we_d;                // engine write strobe
    reg  [8:0] wl_we_a;                // engine write address

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wl_dly  <= 4'd0;
            wl_run  <= 1'b0;
            wl_addr <= 9'd0;
            wl_req  <= 1'b0;
            wl_we_d <= 1'b0;
            wl_we_a <= 9'd0;
        end else if (!eng_ready) begin
            if (!wl_run) begin
                if (wl_dly == 4'd8) begin
                    wl_run  <= 1'b1;
                    wl_addr <= 9'd0;
                    wl_req  <= 1'b1;
                end else begin
                    wl_dly <= wl_dly + 4'd1;
                end
            end else begin
                if (wl_addr == 9'd270) wl_req <= 1'b0;
                else                   wl_addr <= wl_addr + 9'd1;
            end
            // one-cycle-delayed write into the engine (wb_q is registered)
            wl_we_d <= wl_req;
            wl_we_a <= wl_addr;
        end else begin
            wl_req  <= 1'b0;
            wl_we_d <= 1'b0;
            wl_run  <= 1'b0;
        end
    end

    // ================================================================ engine
    wire        eng_start, eng_busy, eng_all_done;
    wire [2:0]  eng_cls;
    wire [75:0] eng_score;
    wire [47:0] eng_acc;
    wire [2:0]  eng_layer, eng_state;
    wire [12:0] eng_beats;
    wire        eng_wload_busy;

    reg         frame_we;
    reg  [11:0] frame_addr;
    reg         frame_bit;

    // ================================================================ D5: FRAME double buffer
    // cap_bank is the bank the engine may read (the "active" frame).  The capture path
    // always writes ~cap_bank, so a frame that arrives while the engine is busy lands in
    // the spare bank and the running inference keeps reading its own bank (the engine
    // latches cap_bank into frm_rd_bank_r on `start`).  Only a *good* frame (CRC checked
    // by frame_rx -> frx_fvalid) publishes the new bank.
    reg         cap_bank;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)          cap_bank <= 1'b0;
        else if (frx_fvalid) cap_bank <= ~cap_bank;
    end

    bnn_engine u_eng (
        .clk(clk), .rst_n(rst_n),
        .start(eng_start),
        .busy(eng_busy), .all_done(eng_all_done), .ready(eng_ready),
        .frame_we(frame_we), .frame_addr(frame_addr), .frame_bit(frame_bit),
        .frame_bank(~cap_bank), .frame_rd_bank(cap_bank),
        .w_we(wl_we_d), .w_addr(wl_we_a), .w_data(wb_q),
        .th_we(1'b0), .th_sel(3'd0), .th_ch(3'd0), .th_data(16'd0),
        .cls_o(eng_cls), .score_o(eng_score), .acc_o(eng_acc),
        .layer_o(eng_layer), .state_o(eng_state), .beat_o(eng_beats),
        .wload_busy(eng_wload_busy)
    );

    // ================================================================ PCPI
    wire [31:0] sfr_rdata;
    wire        pcpi_eng_start;
    wire        pcpi_err;

    pcpi_bnn u_pcpi (
        .clk(clk), .rst_n(rst_n),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
        .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .eng_start(pcpi_eng_start),
        .eng_busy(eng_busy), .eng_ready(eng_ready),
        .score_i(eng_score),
        .status_i(sfr_status),          // bnn.wait returns the STATUS word
        .err_pulse(pcpi_err)
    );

    // ================================================================ SFR
    wire        sfr_start;
    wire        tx_go;
    wire [7:0]  class_o, flags_o, fps_o, frames_o, thresh_o;
    wire [15:0] margin_o, errors_o;
    wire [15:0] frx_seq;
    wire [15:0] frx_nok, frx_nerr;
    wire [15:0] frx_popcnt;

    sfr_bnn #(
        .MODEL_ID(MODEL_ID), .FPS_WIN(FPS_WIN)
    ) u_sfr (
        .clk(clk), .rst_n(rst_n),
        .sel(sel_sfr), .we(sfr_we), .addr(mem_addr[5:0]), .wdata(mem_wdata),
        .rdata(sfr_rdata),
        .ctrl_start(sfr_start), .eng_start(eng_start),
        .eng_busy(eng_busy), .eng_done(eng_all_done), .eng_ready(eng_ready),
        .score_i(eng_score), .cls_i(eng_cls),
        .frame_new(frx_fvalid), .crc_bad(frx_err_pulse), .frame_seq(frx_seq),
        .pcpi_err(pcpi_err),
        .tx_go(tx_go), .class_o(class_o), .margin_o(margin_o),
        .errors_o(errors_o), .fps_o(fps_o), .frames_o(frames_o),
        .flags_o(flags_o), .thresh_o(thresh_o), .status_o(sfr_status)
    );

    assign eng_start = pcpi_eng_start | sfr_start;

    // ================================================================ CPU read mux
    // IMEM/DMEM/WBRAM are block RAM: the data is registered in the cycle the address
    // is presented, so it is valid in the cycle mem_ready is high (2-cycle access).
    // The SFR is a register file and answers combinationally in the same cycle.
    assign mem_rdata = sel_imem ? imem_q :
                       sel_dmem ? dmem_q :
                       sel_sfr  ? sfr_rdata :
                       sel_wb   ? wb_q : 32'hDEAD_0000;

    // ================================================================ UART + frame
    wire [7:0] rx_byte;
    wire       rx_valid;
    wire [7:0] tx_byte;
    wire       tx_start, uart_tx_busy, tx_done;
    wire       ftx_busy;

    uart_rx #(.DIV(BAUD_DIV)) u_urx (
        .clk(clk), .rst_n(rst_n), .rx(pc_uart_rxd),
        .data(rx_byte), .valid(rx_valid)
    );

    uart_tx #(.DIV(BAUD_DIV)) u_utx (
        .clk(clk), .rst_n(rst_n),
        .start(tx_start), .data(tx_byte),
        .tx(pc_uart_txd), .busy(uart_tx_busy), .done(tx_done)
    );

    frame_rx u_frx (
        .clk(clk), .rst_n(rst_n),
        .b(rx_byte), .bvalid(rx_valid),
        .fvalid(frx_fvalid), .popcnt(frx_popcnt), .seq(frx_seq),
        .n_ok(frx_nok), .n_err(frx_nerr)
    );

    // any frame_rx error (header or CRC) -> SFR crc_err + ERRORS++ (one cycle)
    reg [15:0] frx_nerr_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) frx_nerr_q <= 16'd0;
        else        frx_nerr_q <= frx_nerr;
    end
    assign frx_err_pulse = (frx_nerr != frx_nerr_q);

    // ---- result frame snapshot ----
    // frame_tx reads cls/s0..s2/flags/fps/seq combinationally while it shifts the
    // 10 bytes out, so the payload must be latched when the frame is triggered
    // (a CLASS write) instead of being read live: the firmware clears the result
    // registers and rewrites the scores long before the last byte leaves the pin.
    reg        tx_req;
    reg [7:0]  tx_cls, tx_s0, tx_s1, tx_s2, tx_flags, tx_fps, tx_seq;
    wire       tx_fire = tx_req && !ftx_busy;

    // ---- D2: 8-bit result-frame score = saturating Q6 / 1024 ------------------
    // s = clamp(score_q >>> 10, -128, 127)  (arithmetic shift, then int8 saturation).
    // The full 19-bit Q6 value is still available through SFR SCORE0..3 and bnn.acc.
    function [7:0] sat8;
        input signed [18:0] q;
        reg   signed [8:0]  t;
        begin
            t = q >>> 10;
            if      (t >  9'sd127) sat8 = 8'd127;
            else if (t < -9'sd128) sat8 = 8'h80;      // -128
            else                   sat8 = t[7:0];
        end
    endfunction

    wire [7:0] cur_s0  = sat8(eng_score[0*19 +: 19]);
    wire [7:0] cur_s1  = sat8(eng_score[1*19 +: 19]);
    wire [7:0] cur_s2  = sat8(eng_score[2*19 +: 19]);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_req   <= 1'b0;
            tx_cls   <= 8'd0;
            tx_s0    <= 8'd0;
            tx_s1    <= 8'd0;
            tx_s2    <= 8'd0;
            tx_flags <= 8'd0;
            tx_fps   <= 8'd0;
            tx_seq   <= 8'd0;
        end else if (tx_go) begin
            tx_cls   <= class_o;
            tx_s0    <= cur_s0;
            tx_s1    <= cur_s1;
            tx_s2    <= cur_s2;
            tx_flags <= flags_o;
            tx_fps   <= frames_o;
            tx_seq   <= frx_seq[7:0];
            tx_req   <= 1'b1;
        end else if (tx_fire) begin
            tx_req   <= 1'b0;
        end
    end

    frame_tx u_ftx (
        .clk(clk), .rst_n(rst_n),
        .start(tx_fire),
        .cls(tx_cls),
        .s0(tx_s0),
        .s1(tx_s1),
        .s2(tx_s2),
        .flags(tx_flags),
        .fps(tx_fps),
        .seq8(tx_seq),
        .tx_byte(tx_byte), .tx_start(tx_start), .tx_done(tx_done),
        .busy(ftx_busy)
    );

    // ================================================================ payload capture
    // 517 B: A5 | 10 | 00 | 02 | payload[512] | CRC8.  Each payload byte carries 8
    // pixels, MSB first (src/pc/preprocess.py: np.packbits, row major, MSB
    // first) -> pixel 8*i+j = payload byte i bit (7-j), FRAME address = 8*i+j.
    //
    // The header is re-parsed here only to know which payload byte is on the wire.
    // The state sequence and accept rules are identical to frame_rx (including the
    // final CRC byte being consumed), so the two parsers can never drift apart.
    // The CRC itself is checked by frame_rx: a bad frame never raises fvalid, so it
    // never publishes its bank and never raises frame_ready.  With the D5 double
    // buffer the payload bytes go to ~cap_bank, so even a bad frame (or a good frame
    // that arrives mid-inference) cannot corrupt the frame the engine is reading.
    // A byte needs 10*BAUD_DIV >= 10 cycles on the wire and the serialiser needs 8,
    // so an overrun is impossible for BAUD_DIV >= 1.
    localparam [2:0] C_SOF = 3'd0, C_CMD = 3'd1, C_LENL = 3'd2,
                     C_LENH = 3'd3, C_PAY = 3'd4, C_CRC = 3'd5;

    reg [2:0]  cap_st;
    reg [8:0]  cap_cnt;
    reg        fw_busy;
    reg [3:0]  fw_bit;
    reg [7:0]  fw_byte;
    reg [11:0] fw_base;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cap_st     <= C_SOF;
            cap_cnt    <= 9'd0;
            fw_busy    <= 1'b0;
            fw_bit     <= 4'd0;
            fw_byte    <= 8'd0;
            fw_base    <= 12'd0;
            frame_we   <= 1'b0;
            frame_addr <= 12'd0;
            frame_bit  <= 1'b0;
        end else begin
            frame_we <= 1'b0;

            // ---- bit serialiser: 8 bits per payload byte, MSB first ----
            if (fw_busy) begin
                frame_we   <= 1'b1;
                frame_addr <= fw_base + {9'd0, fw_bit[2:0]};
                frame_bit  <= fw_byte[7 - fw_bit[2:0]];
                fw_bit     <= fw_bit + 4'd1;
                if (fw_bit[2:0] == 3'd7) fw_busy <= 1'b0;
            end

            // ---- 517 B downlink parser ----
            if (rx_valid) begin
                case (cap_st)
                C_SOF:  if (rx_byte == 8'hA5) cap_st <= C_CMD;
                C_CMD:  cap_st <= (rx_byte == 8'h10) ? C_LENL : C_SOF;
                C_LENL: cap_st <= (rx_byte == 8'h00) ? C_LENH : C_SOF;
                C_LENH: begin
                            if (rx_byte == 8'h02) begin
                                cap_cnt <= 9'd0;
                                cap_st  <= C_PAY;
                            end else begin
                                cap_st <= C_SOF;
                            end
                        end
                C_PAY:  begin
                            fw_byte <= rx_byte;
                            fw_base <= {cap_cnt, 3'b000};
                            fw_bit  <= 4'd0;
                            fw_busy <= 1'b1;
                            if (cap_cnt == 9'd511) cap_st <= C_CRC;
                            else                   cap_cnt <= cap_cnt + 9'd1;
                        end
                C_CRC:  cap_st <= C_SOF;
                default: cap_st <= C_SOF;
                endcase
            end
        end
    end

    // ================================================================ LED / VGA
    reg [23:0] blink;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) blink <= 24'd0;
        else        blink <= blink + 24'd1;
    end

    assign led = { blink[23], (errors_o != 16'd0), flags_o[0], eng_busy };

    // ---- T5: VGA 640x480@60 self-test + result overlay (no framebuffer) ----
    // vga_simple draws the picture from the raster counters, so it needs no BRAM
    // and works with no CPU/UART traffic ("pull the serial cable and it still
    // displays").  The 25 MHz pixel tick is generated inside vga_simple (/2 of
    // clk), which keeps this file to an instance plus wiring only.
    //   busy = engine busy, done = all_done pulse or the SFR sticky DONE,
    //   err  = SFR sticky err (= STATUS.bit2: CRC bad / PCPI reject).  The 16-bit
    //          ERRORS counter is deliberately NOT used: it only clears on reset,
    //          so ORing it in would leave ERR lit forever, even after CTRL.CLR.
    vga_simple u_vga (
        .clk        (clk),
        .rst_n      (rst_n),
        .pix_clk_en (1'b1),
        .cls        (eng_cls),
        .score      (eng_score),
        .busy       (eng_busy),
        .done       (eng_all_done | sfr_status[1]),
        .err        (sfr_status[2]),
        .vga_hs     (vga_hs),
        .vga_vs     (vga_vs),
        .vga_rgb    (vga_rgb)
    );

    wire unused_ok = &{1'b0, trap, mem_instr, eng_layer[0], eng_state[0],
                       eng_beats[0], eng_wload_busy, margin_o[0], thresh_o[0],
                       frx_nok[0], frx_popcnt[0], wl_we_a[0], uart_tx_busy, fps_o[0]};

endmodule
