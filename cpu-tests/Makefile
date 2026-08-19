# cpu-tests — bare-metal MIPS III/IV CPU test suite for IRIS.
#
#   make            build cputest.elf
#   make run        build, then run it under IRIS via --load-elf
#   make dis        disassemble the built binary
#   make image      build a bootable SGI disk image (volume header + ELF)
#   make clean

# `include` comes first, and toolchain.mk defines a target — so pin the default
# goal before it, or `make` with no arguments would just run toolchain-check.
.DEFAULT_GOAL := all

include toolchain.mk

BUILD   := build
TARGET  := $(BUILD)/cputest.elf

# MIPS III baseline: the binary must run on an R4400. MIPS IV instructions are
# emitted only inside explicit `.set mips4` regions in the mips4/ tests, which
# are gated at runtime on PRId — so a stray MIPS IV opcode from the compiler
# would be a silent R4400 failure. -march=mips3 makes that impossible.
#
# -mabi=n32    NOT o32. Under o32 a `long long` lives in an even/odd register
#              PAIR, so an asm operand like "=r"(u64) binds only the first
#              register of the pair — every 64-bit test would silently read
#              half a result. n32 keeps 64-bit values in single registers,
#              which is what the tests are actually about. Still ELF32 MSB, so
#              --load-elf and the PROM both accept it.
# -mno-abicalls -fno-pic   no GOT, no PIC prologues: we link at a fixed address.
# -G0          nothing in .sdata/.sbss, so no $gp-relative addressing.
# -msoft-float the *compiler* must not emit FP instructions on its own; the FPU
#              tests drive the FPU by hand with explicit asm (inside `.set
#              hardfloat`). Without this GCC would emit its own FP code that
#              assumes an FR/FCSR state the tests are actively changing.
ARCHFLAGS := -march=mips3 -mabi=n32 -EB -mno-abicalls -fno-pic -G0 -msoft-float
CFLAGS    := $(ARCHFLAGS) -ffreestanding -nostdlib -nostdinc \
             -fno-builtin -fno-stack-protector -fno-strict-aliasing \
             -fno-delete-null-pointer-checks -fomit-frame-pointer \
             -O1 -g -std=gnu11 \
             -Wall -Wextra -Werror -Wno-unused-parameter \
             -Iharness
ASFLAGS   := $(ARCHFLAGS) -ffreestanding -nostdlib -Iharness -g -Wa,--fatal-warnings
LDFLAGS   := -EB -T harness/link.ld -nostdlib --no-warn-mismatch

HARNESS_C := harness/console.c harness/testlib.c
HARNESS_S := harness/start.S
TEST_C    := $(sort $(wildcard tests/*/*.c))
TEST_S    := $(sort $(wildcard tests/*/*.S))
REG_C     := harness/tests.c

SRCS_C := $(HARNESS_C) $(REG_C) $(TEST_C)
SRCS_S := $(HARNESS_S) $(TEST_S)
OBJS   := $(patsubst %.c,$(BUILD)/%.o,$(SRCS_C)) $(patsubst %.S,$(BUILD)/%.o,$(SRCS_S))
DEPS   := $(OBJS:.o=.d)

# No libgcc. Under n32 every 64-bit operation the C code performs — including
# the division in con_udec — is a native instruction, so no helpers are needed.
# That matters because the Debian cross package ships only an o32 libgcc.a;
# linking it into an n32 image would be silently wrong. If a helper reference
# ever does appear, the link fails loudly rather than pulling in the wrong ABI.
LIBGCC :=

.PHONY: all clean run dis syms image check-size vectors
all: $(TARGET)

# Regenerate the IEEE-754 expectation tables in tests/fpu/. Needs python3, and
# nothing else does — the generated files are checked in so that building the
# suite stays a cross-toolchain-only affair. --check cross-checks the exact
# rational arithmetic against the host FPU before anything is written.
vectors:
	cd $(CURDIR) && python3 gen/fpvectors.py --check
	cd $(CURDIR) && python3 gen/fpvectors.py

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -MMD -MP -c -o $@ $<

$(TARGET): $(OBJS) harness/link.ld
	@mkdir -p $(dir $@)
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(LIBGCC)
	@$(MAKE) --no-print-directory check-size

# A stray .got, or a PIC prologue, means the -mno-abicalls contract broke and
# the binary would fault the moment it runs unrelocated.
check-size: $(TARGET)
	@$(READELF) -S $(TARGET) | grep -qE '\.got|\.dynamic' && { \
	   echo "error: $(TARGET) has a GOT/dynamic section — build is not -mno-abicalls clean"; \
	   exit 1; } || true
	@echo "built $(TARGET):"
	@$(READELF) -h $(TARGET) | grep -E 'Class|Data|Machine|Entry|Flags'
	@$(NM) $(TARGET) | grep -E ' (_ftext|_etext|_fbss|_end|_stack_top)$$' | sort

dis: $(TARGET)
	$(OBJDUMP) -d $(TARGET) | less

syms: $(TARGET)
	$(NM) -n $(TARGET)

clean:
	rm -rf $(BUILD)

# ── running ──────────────────────────────────────────────────────────────────
IRIS ?= ../target/release/iris

run: $(TARGET)
	@run/run-local.sh $(TARGET)

# ── bootable image ───────────────────────────────────────────────────────────
MKVH ?= ../target/release/mkvh

image: $(BUILD)/cputest.img
$(BUILD)/cputest.img: $(TARGET)
	$(MKVH) build $@ --bootfile cputest cputest=$(TARGET)
	$(MKVH) dump $@

-include $(DEPS)
