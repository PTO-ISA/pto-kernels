
.section linxboot_text, "ax"

.extern _linx_start

_start:
  HL.BSTART.STD CALL, _linx_start, ra=_end
  C.BSTOP
_end:
  BSTART.STD
  addi zero, 0x5e, ->x1
  acrc 1
  C.BSTOP
