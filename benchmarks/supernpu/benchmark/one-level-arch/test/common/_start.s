.global  _start
  .type   _start,@function
  .text
_start:
  HL.BSTART.STD CALL, main, ra=_end
  C.BSTOP
_end:
  BSTART.STD
  # Linx Linux syscall ABI: a0 keeps main's status; a7 carries exit_group=94.
  addi zero, 0x5e, ->a7
  acrc 1
  C.BSTOP
