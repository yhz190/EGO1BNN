/* bringup.c -- D3 CPU bring-up firmware
 *
 * Prints "CPU alive" on UART (SFR 0x10040) and cycles the 4 LEDs (SFR 0x10048).
 * Echoes any byte received on UART back to the PC (proves RX + TX).
 * No BNN instructions here on purpose: this build uses ENABLE_PCPI=0.
 */
#include <stdint.h>

#define SFR(base, off) (*(volatile uint32_t *)((base) + (off)))
#define SFR_BASE   0x00010000u
#define SFR_STATUS SFR(SFR_BASE, 0x04)
#define SFR_TXDATA SFR(SFR_BASE, 0x40)
#define SFR_TXBUSY SFR(SFR_BASE, 0x40)
#define SFR_RXDATA SFR(SFR_BASE, 0x44)
#define SFR_LED    SFR(SFR_BASE, 0x48)
#define SFR_RXCNT  SFR(SFR_BASE, 0x4C)

static void putc_(char c)
{
    while (SFR_STATUS & 1u) { }      /* wait while tx busy */
    SFR_TXDATA = (uint32_t)(unsigned char)c;
}

static void puts_(const char *s)
{
    while (*s) putc_(*s++);
}

int main(void)
{
    volatile uint32_t i;
    uint32_t led = 1u;
    uint32_t seen;
    uint32_t last_cnt = 0;

    puts_("CPU alive\r\n");

    for (;;) {
        SFR_LED = led;
        led = (led << 1) & 0xFu;
        if (!led) led = 1u;

        for (i = 0; i < 800000u; i++) { }      /* slow heartbeat */

        /* echo any received byte */
        seen = SFR_RXCNT;
        if (seen != last_cnt) {
            last_cnt = seen;
            putc_((char)(SFR_RXDATA & 0xFFu));
        }
    }
}
