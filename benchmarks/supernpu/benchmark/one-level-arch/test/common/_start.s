.global  _start
  .type   _start,@function
  .text
_start:
  HL.BSTART.STD CALL, main, ra=_end
  C.BSTOP
_end:
  BSTART.STD
  addi zero, 0x5e, ->x1
  acrc 1
  C.BSTOP
