OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 8K
}

SECTIONS
{
  .text : {
    *(.text._start)
    *(.text*)
  } > RAM

  .rodata : { *(.rodata*) } > RAM
  .data   : { *(.data*)   } > RAM
  .bss    : { *(.bss*) *(COMMON) } > RAM

  . = ALIGN(4);
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}