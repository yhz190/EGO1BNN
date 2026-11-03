/* link_bnn.ld -- top_bnn firmware layout (PicoRV32, 4 KB IMEM + 4 KB DMEM, D6)
 *
 *   0x0000_0000  4 KB  IMEM (ROM, $readmemh fw.hex)   .text + .rodata
 *   0x0000_2000  4 KB  DMEM (read/write)              .data + .bss + stack
 *   0x0000_2FFC        stack top (_stack_top), grows down into DMEM
 *
 * D6: IMEM/DMEM are 1024 x 32 each so the BRAM budget is 5 x RAMB36 + 2 x RAMB18
 * = 6.0 equivalents (T6).  The firmware is built with no initialised globals, so
 * .data/.bss stay empty and `objcopy -O binary` produces the IMEM image only
 * (the build script asserts this).  The image is ~0.5 KB << 4 KB.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  IMEM (rx)  : ORIGIN = 0x00000000, LENGTH = 4K
  DMEM (rw)  : ORIGIN = 0x00002000, LENGTH = 4K
}

SECTIONS
{
  .text : {
    *(.text._start)
    *(.text*)
    . = ALIGN(4);
  } > IMEM

  .rodata : {
    *(.rodata*)
    *(.srodata*)
    . = ALIGN(4);
  } > IMEM

  .data : {
    *(.data*)
    *(.sdata*)
    . = ALIGN(4);
  } > DMEM

  .bss : {
    *(.bss*)
    *(.sbss*)
    *(COMMON)
    . = ALIGN(4);
  } > DMEM

  /* last word of DMEM: the first push writes 0x2FF8, so 0x2FFC itself stays free */
  _stack_top = ORIGIN(DMEM) + LENGTH(DMEM) - 4;
}
