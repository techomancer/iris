# toolchain.mk — locate the MIPS cross toolchain.
#
# Override CROSS on the command line for a different prefix:
#   make CROSS=mips64-elf-
#
# Search order:
#   1. $CROSS if set
#   2. mips-linux-gnu- on PATH (Debian/Ubuntu: gcc-mips-linux-gnu)
#   3. ~/.local/opt/mips-linux-gnu — a rootless `dpkg -x` install, which is
#      what an unattended session without sudo ends up with. See docs/toolchain.md.

LOCAL_TC := $(HOME)/.local/opt/mips-linux-gnu

ifeq ($(origin CROSS), undefined)
  ifneq ($(shell command -v mips-linux-gnu-gcc 2>/dev/null),)
    CROSS := mips-linux-gnu-
  else ifneq ($(wildcard $(LOCAL_TC)/usr/bin/mips-linux-gnu-gcc),)
    CROSS := $(LOCAL_TC)/usr/bin/mips-linux-gnu-
    # binutils from the .deb links against its own libbfd/libopcodes.
    export LD_LIBRARY_PATH := $(LOCAL_TC)/usr/lib/x86_64-linux-gnu:$(LD_LIBRARY_PATH)
  else
    CROSS := mips-linux-gnu-
  endif
endif

CC      := $(CROSS)gcc
LD      := $(CROSS)ld
OBJCOPY := $(CROSS)objcopy
OBJDUMP := $(CROSS)objdump
READELF := $(CROSS)readelf
NM      := $(CROSS)nm

.PHONY: toolchain-check
toolchain-check:
	@command -v $(CC) >/dev/null 2>&1 || { \
	  echo "error: $(CC) not found."; \
	  echo "  Debian/Ubuntu:  sudo apt-get install gcc-mips-linux-gnu binutils-mips-linux-gnu"; \
	  echo "  No root:        make -C cpu-tests toolchain-local"; \
	  exit 1; }
	@echo "toolchain: $$($(CC) --version | head -1)"
