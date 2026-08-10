ifeq ($(TESTCASE), gelu)
DEFINES += -DDTYPE=$(DTYPE) -DtMs=$(tMs) -DgMs=$(gMs) -DApproximate=$(Approximate)
TARGET = $(ELF_HEAD)_$(TESTCASE)_Approximate_$(Approximate)_DType$(DTYPE)_tM$(tMs)_SHAPE$(SHAPE_NAME).elf
endif

# DEFINES += -DRES_CHECK

SRC_FILE +=  $(TEST_ROOT)/$(CASE_SRC_DIR)/$(TESTCASE).cpp
include ../../../common/Makefile.common