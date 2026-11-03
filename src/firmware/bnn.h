/* bnn.h —— SFR 映射 + Custom-0 指令封装（方案 v2.2 §3.4 / §5.1） */
#ifndef BNN_H
#define BNN_H

#include <stdint.h>

/* ---------- SFR（0x0001_0000） ---------- */
#define SFR_BASE      0x00010000u
#define SFR(off)      (*(volatile uint32_t *)(SFR_BASE + (off)))

#define SFR_CTRL      SFR(0x00)
#define SFR_STATUS    SFR(0x04)
#define SFR_THRESH    SFR(0x08)
#define SFR_CLASS     SFR(0x0C)
#define SFR_SCORE0    SFR(0x10)
#define SFR_SCORE1    SFR(0x14)
#define SFR_SCORE2    SFR(0x18)
#define SFR_FPS       SFR(0x1C)
#define SFR_SEQ       SFR(0x20)
#define SFR_MODEL_ID  SFR(0x24)
#define SFR_MARGIN    SFR(0x28)
#define SFR_ERRORS    SFR(0x2C)
#define SFR_SCORE3    SFR(0x30)

#define ST_BUSY       0x01u
#define ST_DONE       0x02u
#define ST_ERR        0x04u
#define ST_FRAME_RDY  0x08u
#define ST_CRC_ERR    0x10u

/* ---------- 存储映射 ---------- */
#define FRAME_BASE    0x00030000u
#define WBRAM_BASE    0x00020000u
#define GALLERY_BASE  0x00040000u

/* ---------- Custom-0 指令 ----------
 * .insn r opcode, func3, func7, rd, rs1, rs2
 * bnn.conv : f3=0  f7=起始层(0~4)  rs1=输入  rs2=权重
 * bnn.wait : f3=1  rd=状态
 * bnn.acc  : f3=2  f7=类别(0~3)    rd=得分
 */
#define BNN_CONV(start_layer, in_base, w_base)                       \
    __asm__ volatile (".insn r 0x0B, 0, %2, x0, %0, %1"              \
                      :: "r"(in_base), "r"(w_base), "i"(start_layer) \
                      : "memory")

static inline uint32_t bnn_wait(void)
{
    uint32_t r;
    __asm__ volatile (".insn r 0x0B, 1, 0, %0, x0, x0" : "=r"(r) : : "memory");
    return r;
}

#define BNN_ACC(N)                                                    \
    ({ uint32_t r_;                                                   \
       __asm__ volatile (".insn r 0x0B, 2, " #N ", %0, x0, x0"        \
                         : "=r"(r_) : : "memory");                    \
       r_; })

#endif /* BNN_H */