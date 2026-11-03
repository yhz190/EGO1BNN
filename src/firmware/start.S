    .section .text._start
    .globl _start
    .type _start, @function
_start:
    la      sp, _stack_top          /* 栈顶 = 0x2000 */
    call    main
1:  j       1b                      /* main 返回后原地循环 */
    .size _start, . - _start