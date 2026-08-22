# PTO ISA 0.58.3 Local-CELL matrix smoke.

TYPE ?= GMMA

ifneq ($(TYPE),GMMA)
$(error Unsupported TYPE=$(TYPE); the pre-CELL MASK/HIF4/A16W4 paths are archived under status/legacy)
endif

SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/matmul_gmma.cpp
DEFINES += -DglobM=$(M) -DglobN=$(N) -DglobK=$(K)
DEFINES += -DtilM=$(tM) -DtilN=$(tN) -DtilK=$(tK)
ifneq ($(MODE),)
DEFINES += -D$(MODE)
endif
TARGET = $(ELF_HEAD)/$(TESTCASE)_CELL_$(MODE)_M$(M)_N$(N)_K$(K)_tM$(tM)_tN$(tN)_tK$(tK).elf

include ../../common/Makefile.common
