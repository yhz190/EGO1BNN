/* main_bnn.c -- top_bnn firmware (D4/G1 integration, v2.2-patch2 + D2/D5/D6)
 *
 * Flow (task E):
 *   init -> DMEM 4 KB map self test -> SFR self test -> wait STATUS.frame_ready
 *        -> launch one full frame -> block on bnn.wait
 *        -> read the 4 scores (PCPI bnn.acc AND the SFR map) -> classify (junk first,
 *           then the Q6 margin gate) -> write CLASS (which makes frame_tx send the
 *           10 B result frame) -> clear done.
 *
 * The start goes through `bnn.conv` because 09_完整方案 v2.2 sec.5.3 makes it the
 * single start path ("bnn.conv 是引擎唯一启动入口").  SFR CTRL.START is implemented
 * in sfr_bnn as an equivalent alias, but the canonical firmware path is the
 * instruction.  See sim/20261101_top_bnn集成记录.md sec.8.
 *
 * No globals: .data/.bss stay empty, so IMEM alone (fw.hex) is enough to boot and
 * the stack lives in the top word of the 4 KB DMEM (picorv32 STACKADDR = 0x2FFC,
 * _stack_top = 0x2FFC; the first push lands at 0x2FF8).
 */
#include "bnn.h"

/* Self-test failure marker written to CLASS (never a valid class code). */
#define SELFTEST_FAIL 0xEEu

/* D6 DMEM map test patterns: top word of the 4 KB window and the bottom word.
 * They are kept in DMEM so the integration TB can read them back through the
 * hierarchical array (a decode that spilled past 0x2FFF or aliased the low end
 * would corrupt one of them).  DMEM[1] (0x2004) publishes the pass/fail word. */
#define DMEM_TOP_ADDR  0x00002FFCu
#define DMEM_BOT_ADDR  0x00002000u
#define DMEM_RSLT_ADDR 0x00002004u
#define DMEM_TOP_PAT   0x5A5A2FFCu
#define DMEM_BOT_PAT   0xC3C32000u
#define DMEM_PASS_WORD 0x600DBEEFu
#define DMEM_FAIL_WORD 0xDEAD0000u

static int32_t signed_score(uint32_t v)
{
    /* bnn.acc returns the 19 bit Q6 score sign extended to 32 bit */
    return (int32_t)v;
}

int main(void)
{
    uint32_t ok = 1;

    /* ---- 0. DMEM 4 KB map self test (D6) ---------------------------------
     * DMEM is 1024 x 32 at 0x2000..0x2FFF.  The stack pointer starts at 0x2FFC
     * and grows down, so the top word is free for this test.  Writing the top
     * and bottom words and reading them back proves the 10-bit word index does
     * not spill past 0x2FFF nor alias the low end into the high end.
     */
    volatile uint32_t *dmem_top = (volatile uint32_t *)DMEM_TOP_ADDR;
    volatile uint32_t *dmem_bot = (volatile uint32_t *)DMEM_BOT_ADDR;

    *dmem_top = DMEM_TOP_PAT;
    *dmem_bot = DMEM_BOT_PAT;
    if ((*dmem_top != DMEM_TOP_PAT) || (*dmem_bot != DMEM_BOT_PAT)) ok = 0;
    *((volatile uint32_t *)DMEM_RSLT_ADDR) = ok ? DMEM_PASS_WORD : DMEM_FAIL_WORD;

    /* ---- 1. SFR read/write self test -------------------------------------
     * MARGIN (0x28) is the only read/write field with a non-trivial reset value
     * (1536 = 24*64, Q6).  Write a pattern, read it back, restore it.
     * (There is no LED SFR in v2.2 sec.3.4; the LEDs are driven by hardware.)
     */
    SFR_MARGIN = 0x0000A55Au;
    if (SFR_MARGIN != 0x0000A55Au) ok = 0;
    /* acceptance mode: MARGIN=0 -> pure argmax (matches the model exactly);
     * a demo may set a non-zero Q6 margin (old 24 -> 1536) for junk rejection. */
    SFR_MARGIN = 0u;

    /* ---- 2. illegal Custom-0 funct3=7 must not hang the CPU (v2.2 R5) ----
     * pcpi_bnn answers ready with wr=0; if that were wrong this instruction would
     * stall cpu_state_ld_rs1 forever and no result frame would ever be sent.
     */
    __asm__ volatile (".insn r 0x0B, 7, 0, x0, x0, x0" ::: "memory");

    if (!ok) {
        SFR_CLASS = SELFTEST_FAIL;      /* observable failure marker */
        SFR_CTRL  = 0x02u;
    }

    /* ---- 3. inference loop ---------------------------------------------- */
    uint32_t last_seq = 0xFFFFFFFFu;
    for (;;) {
        uint32_t st = SFR_STATUS;
        if (!(st & ST_FRAME_RDY)) continue;

        uint32_t seq = SFR_SEQ;
        if (seq == last_seq) continue;
        last_seq = seq;

        /* 3.1 launch one full frame (L1..L5) */
        BNN_CONV(0, FRAME_BASE, WBRAM_BASE);

        /* 3.2 block until the engine is idle (bnn.wait also returns STATUS) */
        while (bnn_wait() & ST_BUSY) { }

        /* 3.3 scores, both read paths (must agree) */
        uint32_t a0 = BNN_ACC(0), a1 = BNN_ACC(1), a2 = BNN_ACC(2), a3 = BNN_ACC(3);
        uint32_t r0 = SFR_SCORE0, r1 = SFR_SCORE1, r2 = SFR_SCORE2, r3 = SFR_SCORE3;

        uint32_t bad = (a0 != r0) || (a1 != r1) || (a2 != r2) || (a3 != r3);

        int32_t q0 = signed_score(a0), q1 = signed_score(a1);
        int32_t q2 = signed_score(a2), q3 = signed_score(a3);

        /* 3.4 decision: junk (class 3) wins outright, then the margin gate */
        uint32_t cls;
        if (q3 > q0 && q3 > q1 && q3 > q2) {
            cls = 3u;
        } else {
            cls = (q0 >= q1 && q0 >= q2) ? 0u : ((q1 >= q2) ? 1u : 2u);

            int32_t top1 = (cls == 0u) ? q0 : ((cls == 1u) ? q1 : q2);
            int32_t top2 = (cls == 0u) ? ((q1 > q2) ? q1 : q2)
                         : (cls == 1u) ? ((q0 > q2) ? q0 : q2)
                                       : ((q0 > q1) ? q0 : q1);
            if ((top1 - top2) < (int32_t)SFR_MARGIN) cls = 3u;
        }

        if (bad) cls = SELFTEST_FAIL;

        /* 3.5 publishing CLASS triggers the 10 B result frame in hardware */
        SFR_CLASS = cls;

        /* 3.6 clear done and let frame_ready fall (it fell on the start) */
        SFR_CTRL = 0x02u;
    }
}
