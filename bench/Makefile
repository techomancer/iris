# bench — the IRIS benchmark suite, bare metal.
#
#   make            build build/irisbench.elf
#   make run        build, then run it under IRIS via --load-elf
#   make golden     regenerate golden/golden.h from a native host build
#   make image      bootable SGI disk image (volume header + ELF)
#   make matrix     run every CPU x engine cell and write a comparison
#   make clean
#
# The toolchain lookup, the console and the startup/exception code are shared
# with cpu-tests rather than duplicated: one MIPS cross-compiler probe, one SCC
# driver, one exception dispatcher. What is not shared is the float ABI (this
# suite wants the FPU; cpu-tests drives it by hand under -msoft-float), the
# link layout (this one needs megabytes of working set) and the runner.

.DEFAULT_GOAL := all

CPUTESTS := ../cpu-tests
include $(CPUTESTS)/toolchain.mk

BUILD  := build
TARGET := $(BUILD)/irisbench.elf

# -march=mips3     an R4400 must be able to run it; the R5000 cells run the
#                  same binary, and anything MIPS IV would be a silent R4400
#                  Reserved Instruction rather than a loud build error.
# -mabi=n32        64-bit values in single registers. Still ELF32 MSB, so
#                  --load-elf and the PROM both accept it.
# -mhard-float     the opposite of cpu-tests, and the point: these kernels are
#                  meant to hammer the FPU through ordinary compiled code.
# -ffp-contract=off  no multiply-add fusion. MIPS III has no FMA, so the guest
#                  could not fuse anyway — but the golden generator runs on a
#                  host that can, and a contracted result differs in the last
#                  ulp from an uncontracted one.
ARCHFLAGS := -march=mips3 -mabi=n32 -EB -mno-abicalls -fno-pic -G0 -mhard-float
CFLAGS    := $(ARCHFLAGS) -ffreestanding -nostdlib -nostdinc \
             -fno-builtin -fno-stack-protector -fno-strict-aliasing \
             -fno-delete-null-pointer-checks -fomit-frame-pointer \
             -ffp-contract=off -fexcess-precision=standard \
             -O2 -g -std=gnu11 \
             -Wall -Wextra -Werror -Wno-unused-parameter \
             -Iharness -Igolden -I$(CPUTESTS)/harness
ASFLAGS   := $(ARCHFLAGS) -ffreestanding -nostdlib -I$(CPUTESTS)/harness -Iharness \
             -g -Wa,--fatal-warnings
LDFLAGS   := -EB -T harness/link.ld -nostdlib --no-warn-mismatch

HARNESS_C := harness/benchlib.c harness/string.c harness/groups.c harness/main.c \
             $(CPUTESTS)/harness/console.c
HARNESS_S := $(CPUTESTS)/harness/start.S harness/tlbasm.S
KERNEL_C  := $(sort $(wildcard kernels/*.c))

OBJS := $(patsubst %.c,$(BUILD)/%.o,$(notdir $(HARNESS_C))) \
        $(patsubst %.S,$(BUILD)/%.o,$(notdir $(HARNESS_S))) \
        $(patsubst kernels/%.c,$(BUILD)/kernels/%.o,$(KERNEL_C))
DEPS := $(OBJS:.o=.d)

# No libgcc, for the same reason cpu-tests has none: the Debian cross package
# ships only an o32 copy, and linking that into an n32 image is silently wrong.
# Under n32 with hard float every operation these kernels perform is a native
# instruction. If a helper reference ever appears the link fails loudly.
LIBGCC :=

.PHONY: all clean run dis syms image golden hostbench matrix bench report check-size
all: $(TARGET)

$(BUILD)/%.o: harness/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/%.o: $(CPUTESTS)/harness/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/kernels/%.o: kernels/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/%.o: harness/%.S
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -MMD -MP -c -o $@ $<

$(BUILD)/%.o: $(CPUTESTS)/harness/%.S
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -MMD -MP -c -o $@ $<

$(TARGET): $(OBJS) harness/link.ld
	@mkdir -p $(dir $@)
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(LIBGCC)
	@$(MAKE) --no-print-directory check-size

check-size: $(TARGET)
	@$(READELF) -S $(TARGET) | grep -qE '\.got|\.dynamic' && { \
	   echo "error: $(TARGET) has a GOT/dynamic section — not -mno-abicalls clean"; \
	   exit 1; } || true
	@echo "built $(TARGET):"
	@$(READELF) -h $(TARGET) | grep -E 'Class|Data|Machine|Entry|Flags'
	@$(NM) $(TARGET) | grep -E ' (_ftext|_etext|_fbss|_end|_work_start)$$' | sort

dis: $(TARGET)
	$(OBJDUMP) -d $(TARGET) | less

syms: $(TARGET)
	$(NM) -n $(TARGET)

clean:
	rm -rf $(BUILD)

# ── the oracle ───────────────────────────────────────────────────────────────
# Native build of the same kernels. -O1 and no vectoriser: see gen/golden.c.
HOSTCC     ?= cc
HOSTCFLAGS := -O1 -std=gnu11 -DBENCH_HOST -Iharness -Igolden \
              -fno-tree-vectorize -ffp-contract=off -fexcess-precision=standard \
              -Wall -Wextra -Wno-unused-parameter
# sys.c is excluded: a TLB refill and a cache writeback have no host analogue,
# and those kernels carry no golden checksum for exactly that reason.
GOLDEN_SRC := gen/golden.c gen/hostplat.c harness/groups.c $(filter-out kernels/sys.c,$(KERNEL_C))

golden: $(BUILD)/golden-gen
	@$(BUILD)/golden-gen > golden/golden.h.new && mv golden/golden.h.new golden/golden.h
	@echo "wrote golden/golden.h"

$(BUILD)/golden-gen: $(GOLDEN_SRC) harness/benchlib.h harness/hostshim.h harness/bmath.h golden/golden.h
	@mkdir -p $(BUILD)
	$(HOSTCC) $(HOSTCFLAGS) -o $@ $(GOLDEN_SRC) -lm

# ── the host baseline ────────────────────────────────────────────────────────
# The same kernels and the same runner, built for the machine the emulator runs
# on. Emits the identical machine block, so iris-bench can put a native column
# next to the emulated ones and the ratio between them means something.
#
# -O2 here, not the golden generator's -O1: this one is measuring the host, and
# hobbling the compiler would understate it. Checksums are unaffected — they
# are integers and IEEE-754 arithmetic, neither of which -O touches without
# -ffast-math.
HOSTBENCHFLAGS := -O2 -std=gnu11 -DBENCH_HOST -Iharness -Igolden \
                  -ffp-contract=off -fexcess-precision=standard \
                  -Wall -Wextra -Wno-unused-parameter
HOSTBENCH_SRC  := harness/main.c gen/hostplat.c harness/groups.c \
                  $(filter-out kernels/sys.c,$(KERNEL_C))

hostbench: $(BUILD)/irisbench-host
$(BUILD)/irisbench-host: $(HOSTBENCH_SRC) harness/benchlib.h harness/hostshim.h harness/bmath.h golden/golden.h
	@mkdir -p $(BUILD)
	$(HOSTCC) $(HOSTBENCHFLAGS) -o $@ $(HOSTBENCH_SRC) -lm

# ── running ──────────────────────────────────────────────────────────────────
IRIS ?= ../target/release/iris

run: $(TARGET)
	@run/run-local.sh $(TARGET)

# The matrix lives in iris-bench, not in a shell script: it has to build a
# separate emulator per cell (the CPU model and the JIT are cargo features),
# check that each one really is the CPU it claims, parse the results and write
# a comparison. That is a program, and it needs tests.
matrix: $(TARGET)
	cd .. && cargo build --release --bin iris-bench
	cd .. && ./target/release/iris-bench matrix

# One cell, whatever ../target/release/iris happens to be.
bench: $(TARGET)
	cd .. && cargo build --release --bin iris-bench
	cd .. && ./target/release/iris-bench run --label local

report:
	cd .. && ./target/release/iris-bench report

# ── bootable image ───────────────────────────────────────────────────────────
MKVH ?= ../target/release/mkvh

image: $(BUILD)/irisbench.img
$(BUILD)/irisbench.img: $(TARGET)
	$(MKVH) build $@ --bootfile irisbench irisbench=$(TARGET)
	$(MKVH) dump $@

-include $(DEPS)
