# toolchain.mk — locate the MIPS cross toolchain.
#
# Override CROSS on the command line for a different prefix:
#   make CROSS=mips64-elf-
#
# Search order:
#   1. $CROSS if set
#   2. mips-linux-gnu- on PATH (Debian/Ubuntu: gcc-mips-linux-gnu)
#   3. mips-unknown-linux-gnu- on PATH (Gentoo crossdev, 32-bit target)
#   4. mips64-unknown-linux-gnu- on PATH (Gentoo crossdev, 64-bit target —
#      still produces n32/o32 output via ARCHFLAGS, since -mabi is explicit)
#   5. ~/.local/opt/mips-linux-gnu — a rootless `dpkg -x` install, which is
#      what an unattended session without sudo ends up with. See docs/toolchain.md.
#
# A candidate is accepted only if `$(name)gcc --version` actually runs — a
# broken/dangling shim on PATH (crossdev leaves these behind sometimes) must
# not shadow a working toolchain further down the list.

LOCAL_TC := $(HOME)/.local/opt/mips-linux-gnu

cc_works = $(shell $(1)gcc --version >/dev/null 2>&1 && echo yes)

ifeq ($(origin CROSS), undefined)
  ifeq ($(call cc_works,mips-linux-gnu-),yes)
    CROSS := mips-linux-gnu-
  else ifeq ($(call cc_works,mips-unknown-linux-gnu-),yes)
    CROSS := mips-unknown-linux-gnu-
  else ifeq ($(call cc_works,mips64-unknown-linux-gnu-),yes)
    CROSS := mips64-unknown-linux-gnu-
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
	  echo "  Gentoo:         crossdev --target mips64-unknown-linux-gnu"; \
	  echo "  No root:        make -C cpu-tests toolchain-local"; \
	  exit 1; }
	@echo "toolchain: $$($(CC) --version | head -1)"

# Rootless install: unpack the Debian cross-toolchain .debs into a prefix.
# What an unattended session without sudo has to do, and what the CI job does
# with apt-get instead. The binutils from these packages link against their own
# libbfd/libopcodes, which are not on the default loader path in a rootless
# install — hence the LD_LIBRARY_PATH export above and in env.sh.
TC_DEBS := binutils-mips-linux-gnu binutils-common libbinutils \
           gcc-12-mips-linux-gnu cpp-12-mips-linux-gnu libgcc-12-dev-mips-cross

.PHONY: toolchain-local
toolchain-local:
	@mkdir -p $(LOCAL_TC)/.debs
	cd $(LOCAL_TC)/.debs && apt-get download $(TC_DEBS)
	cd $(LOCAL_TC)/.debs && for d in *.deb; do dpkg -x "$$d" $(LOCAL_TC); done
	@printf '%s\n' \
	  '# Source this to put the local mips-linux-gnu cross toolchain on PATH.' \
	  '_MIPS_PFX="$$(cd "$$(dirname "$${BASH_SOURCE[0]}")" && pwd)"' \
	  'export PATH="$$_MIPS_PFX/usr/bin:$$PATH"' \
	  'export LD_LIBRARY_PATH="$$_MIPS_PFX/usr/lib/x86_64-linux-gnu:$$LD_LIBRARY_PATH"' \
	  > $(LOCAL_TC)/env.sh
	@echo "installed into $(LOCAL_TC) — the Makefile finds it automatically"
