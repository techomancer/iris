# toolchain.mk — the MIPS cross toolchain both bare-metal suites build with.
#
# One compiler builds every guest binary in the repository: cpu-tests, bench,
# and the bench image iris embeds. The benchmark measures rates of compiled
# guest code, so the compiler is part of the workload: GCC 14.2 runs Dhrystone
# in 428 guest instructions per iteration where GCC 16.2.1 needs 313. The
# version is pinned below; toolchain.Containerfile builds it (docs/toolchain.md
# has the commands) and must name the same version. A different compiler still
# builds, for quick local runs:
#   make CROSS=<prefix>
GUEST_GCC_VERSION := 16.2.1
GUEST_CROSS       := mips64-linux-gnu-

# The repository root, for -ffile-prefix-map.
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)

# Both suites compile with this, so their debug information records source
# paths relative to the repository root rather than wherever the checkout
# lives, and two builds of the same sources are byte-identical. The root
# rather than each suite's directory, because bench compiles cpu-tests/harness
# too.
PREFIXMAP := -ffile-prefix-map=$(REPO_ROOT)=.

# Search order when CROSS is not given: the pinned triple first, then the
# names other distributions use (any version; see the warning below).
# A candidate is accepted only if `$(name)gcc --version` actually runs — a
# broken/dangling shim on PATH (crossdev leaves these behind sometimes) must
# not shadow a working toolchain further down the list.
cc_works = $(shell $(1)gcc --version >/dev/null 2>&1 && echo yes)

ifeq ($(origin CROSS), undefined)
  ifeq ($(call cc_works,$(GUEST_CROSS)),yes)
    CROSS := $(GUEST_CROSS)
  else ifeq ($(call cc_works,mips-linux-gnu-),yes)
    CROSS := mips-linux-gnu-
  else ifeq ($(call cc_works,mips-unknown-linux-gnu-),yes)
    CROSS := mips-unknown-linux-gnu-
  else ifeq ($(call cc_works,mips64-unknown-linux-gnu-),yes)
    CROSS := mips64-unknown-linux-gnu-
  else
    CROSS := $(GUEST_CROSS)
  endif
endif

CC      := $(CROSS)gcc
LD      := $(CROSS)ld
OBJCOPY := $(CROSS)objcopy
OBJDUMP := $(CROSS)objdump
READELF := $(CROSS)readelf
NM      := $(CROSS)nm

# A different compiler's binaries are not the ones CI tests, and its benchmark
# figures are not comparable.
CC_VERSION := $(shell $(CC) -dumpfullversion 2>/dev/null)
ifneq ($(CC_VERSION),)
  ifneq ($(CC_VERSION),$(GUEST_GCC_VERSION))
    $(warning $(CC) is GCC $(CC_VERSION), not the pinned $(GUEST_GCC_VERSION); see cpu-tests/docs/toolchain.md)
  endif
endif

.PHONY: toolchain-check
toolchain-check:
	@command -v $(CC) >/dev/null 2>&1 || { \
	  echo "error: $(CC) not found. See $(REPO_ROOT)/cpu-tests/docs/toolchain.md"; \
	  exit 1; }
	@echo "toolchain: $$($(CC) --version | head -1)"
