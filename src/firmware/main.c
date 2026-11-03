/* main.c —— 固件骨架（阶段 D1/D2 用，暂不含真实推理循环） */
#include "bnn.h"

static uint32_t g_last_seq;

static inline uint32_t argmax3(uint32_t a, uint32_t b, uint32_t c)
{
    if (a >= b && a >= c) return 0;
    if (b >= c)           return 1;
    return 2;
}

int main(void)
{
    for (;;) {
        uint32_t st = SFR_STATUS;
        if (st & ST_FRAME_RDY) {
            uint32_t seq = SFR_SEQ;
            if (seq == g_last_seq) continue;
            g_last_seq = seq;

            /* 1) 启动整网（funct7=0） */
            BNN_CONV(0, FRAME_BASE, WBRAM_BASE);

            /* 2) 等完成 */
            while (bnn_wait() & ST_BUSY) { }

            /* 3) 取四类得分（含 junk） */
            uint32_t s0 = BNN_ACC(0);
            uint32_t s1 = BNN_ACC(1);
            uint32_t s2 = BNN_ACC(2);
            uint32_t sj = BNN_ACC(3);

            /* 4) 判决：junk 优先，其次 margin */
            uint32_t cls;
            if (sj > s0 && sj > s1 && sj > s2) {
                cls = 3;                       /* UNKNOWN */
            } else {
                cls = argmax3(s0, s1, s2);
                uint32_t top1 = (cls == 0) ? s0 : (cls == 1 ? s1 : s2);
                uint32_t top2 = (cls == 0) ? (s1 > s2 ? s1 : s2)
                                           : (cls == 1 ? (s0 > s2 ? s0 : s2)
                                                       : (s0 > s1 ? s0 : s1));
                if ((top1 - top2) < SFR_MARGIN) cls = 3;
            }

            /* 5) 写结果 */
            SFR_CLASS = cls;
        }
    }
}