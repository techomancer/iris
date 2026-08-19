/* alu — integer arithmetic, logic, shifts, and the 32-bit/64-bit boundary.
 *
 * The recurring theme is sign extension. On a 64-bit MIPS every 32-bit ALU
 * result is defined as the sign-extension of bit 31 into bits 63:32, and is
 * unpredictable if the operands were not themselves properly sign-extended.
 * An emulator that keeps results in a u32 and zero-extends looks correct until
 * a program compares two "positive" numbers whose bit 31 is set. That is the
 * single highest-yield thing this file tests.
 */

#include "testlib.h"
#include "cp0.h"

/* Force values through registers the compiler cannot constant-fold, so we
 * test the CPU rather than GCC's arithmetic. */
#define OPAQUE(x) ({ __typeof__(x) __v = (x); __asm__ __volatile__("" : "+r"(__v)); __v; })

/* ── 32-bit ops sign-extend into 64 bits ──────────────────────────────────── */

static void t_addu_sign_extends(void)
{
    u64 r;
    u32 a = OPAQUE(0x7FFFFFFFu), b = OPAQUE(1u);
    /* 0x7fffffff + 1 = 0x80000000: bit 31 set, so the 64-bit register must
     * read 0xffffffff80000000, not 0x0000000080000000. */
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "addu %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_EQ(r, 0xFFFFFFFF80000000ull);
}

static void t_addiu_sign_extends(void)
{
    u64 r;
    u32 a = OPAQUE(0x7FFFFFFFu);
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "addiu %0, %1, 1\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0xFFFFFFFF80000000ull);
}

static void t_subu_sign_extends(void)
{
    u64 r;
    u32 a = OPAQUE(0u), b = OPAQUE(0x80000000u);
    /* 0 - 0x80000000 = 0x80000000 in 32 bits — again bit 31 set. */
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "subu %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_EQ(r, 0xFFFFFFFF80000000ull);
}

static void t_sll_sign_extends(void)
{
    u64 r;
    u32 a = OPAQUE(1u);
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "sll %0, %1, 31\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0xFFFFFFFF80000000ull);
}

/* SRL/SRA operate on the low 32 bits and sign-extend the 32-bit result. A
 * register holding a sign-extended negative value must still shift as if it
 * were 32-bit. */
static void t_srl_uses_low32(void)
{
    u64 r;
    u64 a = OPAQUE(0xFFFFFFFF80000000ull);
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "srl %0, %1, 4\n\t.set pop"
                         : "=r"(r) : "r"(a));
    /* low 32 = 0x80000000, >> 4 = 0x08000000, bit 31 clear → zero-extended. */
    CHECK_EQ(r, 0x0000000008000000ull);
}

static void t_sra_uses_low32(void)
{
    u64 r;
    u64 a = OPAQUE(0xFFFFFFFF80000000ull);
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "sra %0, %1, 4\n\t.set pop"
                         : "=r"(r) : "r"(a));
    /* 0x80000000 >>a 4 = 0xf8000000, bit 31 set → sign-extended. */
    CHECK_EQ(r, 0xFFFFFFFFF8000000ull);
}

/* Shift amount is taken mod 32 for the 32-bit variable shifts. */
static void t_variable_shift_masks_to_5_bits(void)
{
    u64 r;
    u32 a = OPAQUE(0x00000001u);
    u32 s = OPAQUE(33u);              /* 33 & 31 == 1 */
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "sllv %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(s));
    CHECK_EQ(r, 2ull);

    s = OPAQUE(64u);                  /* 64 & 31 == 0: no shift */
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "sllv %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(s));
    CHECK_EQ(r, 1ull);
}

/* ── 64-bit shifts ────────────────────────────────────────────────────────── */

static void t_dsll_dsrl_dsra(void)
{
    u64 r, a = OPAQUE(1ull);
    __asm__ __volatile__(".set push; .set mips3\n\tdsll %0, %1, 31\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0x0000000080000000ull);   /* no sign extension: it is 64-bit */

    a = OPAQUE(0x8000000000000000ull);
    __asm__ __volatile__(".set push; .set mips3\n\tdsrl %0, %1, 4\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0x0800000000000000ull);

    __asm__ __volatile__(".set push; .set mips3\n\tdsra %0, %1, 4\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0xF800000000000000ull);
}

/* dsll32/dsrl32/dsra32 add 32 to the shift amount, so `dsll32 x, 0` is a
 * 32-place shift — a classic off-by-32. */
static void t_dshift32_variants(void)
{
    u64 r, a = OPAQUE(1ull);
    __asm__ __volatile__(".set push; .set mips3\n\tdsll32 %0, %1, 0\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0x0000000100000000ull);

    __asm__ __volatile__(".set push; .set mips3\n\tdsll32 %0, %1, 31\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0x8000000000000000ull);

    a = OPAQUE(0x8000000000000000ull);
    __asm__ __volatile__(".set push; .set mips3\n\tdsrl32 %0, %1, 0\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0x0000000080000000ull);

    __asm__ __volatile__(".set push; .set mips3\n\tdsra32 %0, %1, 0\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0xFFFFFFFF80000000ull);
}

static void t_dsllv_masks_to_6_bits(void)
{
    u64 r, a = OPAQUE(1ull);
    u32 s = OPAQUE(65u);              /* 65 & 63 == 1 */
    __asm__ __volatile__(".set push; .set mips3\n\tdsllv %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(s));
    CHECK_EQ(r, 2ull);
}

/* ── overflow: the trapping forms trap, the `u` forms do not ──────────────── */

static void t_add_overflow_traps(void)
{
    u64 r = OPAQUE(0xAAAAAAAAAAAAAAAAull);
    u32 a = OPAQUE(0x7FFFFFFFu), b = OPAQUE(1u);

    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "add %0, %1, %2\n\t.set pop"
                         : "+r"(r) : "r"(a), "r"(b));
    CHECK_EXC(EXC_OV);
    /* The destination must be untouched when the instruction traps. */
    CHECK_EQ(r, 0xAAAAAAAAAAAAAAAAull);
    CHECK_EQ(exc.cause & CAUSE_BD, 0u);
}

static void t_addu_does_not_trap(void)
{
    u64 r;
    u32 a = OPAQUE(0x7FFFFFFFu), b = OPAQUE(1u);
    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "addu %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_NO_EXC();
    CHECK_EQ(r, 0xFFFFFFFF80000000ull);
}

static void t_addi_overflow_traps(void)
{
    u64 r = OPAQUE(0x1234ull);
    u32 a = OPAQUE(0x7FFFFFFFu);
    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "addi %0, %1, 1\n\t.set pop"
                         : "+r"(r) : "r"(a));
    CHECK_EXC(EXC_OV);
    CHECK_EQ(r, 0x1234ull);
}

/* Overflow is computed on the 32-bit sum, so 0x80000000 + 0x80000000 (two
 * negatives summing to zero) also traps. */
static void t_add_overflow_negative_side(void)
{
    u64 r = OPAQUE(7ull);
    u32 a = OPAQUE(0x80000000u), b = OPAQUE(0x80000000u);
    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "add %0, %1, %2\n\t.set pop"
                         : "+r"(r) : "r"(a), "r"(b));
    CHECK_EXC(EXC_OV);
    CHECK_EQ(r, 7ull);
}

/* Adding two operands whose signs differ can never overflow, no matter how
 * large — a guard against an emulator that checks carry-out instead. */
static void t_add_no_overflow_on_mixed_signs(void)
{
    u64 r;
    u32 a = OPAQUE(0x80000000u), b = OPAQUE(0x7FFFFFFFu);
    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "add %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_NO_EXC();
    CHECK_EQ(r, 0xFFFFFFFFFFFFFFFFull);
}

static void t_sub_overflow_traps(void)
{
    u64 r = OPAQUE(0x99ull);
    u32 a = OPAQUE(0x80000000u), b = OPAQUE(1u);
    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "sub %0, %1, %2\n\t.set pop"
                         : "+r"(r) : "r"(a), "r"(b));
    CHECK_EXC(EXC_OV);
    CHECK_EQ(r, 0x99ull);
}

static void t_dadd_overflow_traps(void)
{
    u64 r = OPAQUE(0x55ull);
    u64 a = OPAQUE(0x7FFFFFFFFFFFFFFFull), b = OPAQUE(1ull);
    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "dadd %0, %1, %2\n\t.set pop"
                         : "+r"(r) : "r"(a), "r"(b));
    CHECK_EXC(EXC_OV);
    CHECK_EQ(r, 0x55ull);
}

/* The 32-bit overflow condition must NOT apply to dadd: 0x7fffffff + 1 is a
 * perfectly ordinary 64-bit sum. */
static void t_dadd_no_overflow_at_32bit_boundary(void)
{
    u64 r;
    u64 a = OPAQUE(0x000000007FFFFFFFull), b = OPAQUE(1ull);
    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "dadd %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_NO_EXC();
    CHECK_EQ(r, 0x0000000080000000ull);
}

static void t_daddu_does_not_trap(void)
{
    u64 r;
    u64 a = OPAQUE(0x7FFFFFFFFFFFFFFFull), b = OPAQUE(1ull);
    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "daddu %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_NO_EXC();
    CHECK_EQ(r, 0x8000000000000000ull);
}

static void t_dsub_overflow_traps(void)
{
    u64 r = OPAQUE(0x11ull);
    u64 a = OPAQUE(0x8000000000000000ull), b = OPAQUE(1ull);
    exc_clear();
    __asm__ __volatile__(".set push; .set mips3\n\t"
                         "dsub %0, %1, %2\n\t.set pop"
                         : "+r"(r) : "r"(a), "r"(b));
    CHECK_EXC(EXC_OV);
    CHECK_EQ(r, 0x11ull);
}

/* ── logic ────────────────────────────────────────────────────────────────── */

static void t_logical_ops_are_64bit(void)
{
    u64 r;
    u64 a = OPAQUE(0xF0F0F0F0F0F0F0F0ull), b = OPAQUE(0xFF00FF00FF00FF00ull);

    __asm__ __volatile__(".set push; .set mips3\n\tand %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_EQ(r, 0xF000F000F000F000ull);

    __asm__ __volatile__(".set push; .set mips3\n\tor %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_EQ(r, 0xFFF0FFF0FFF0FFF0ull);

    __asm__ __volatile__(".set push; .set mips3\n\txor %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_EQ(r, 0x0FF00FF00FF00FF0ull);

    __asm__ __volatile__(".set push; .set mips3\n\tnor %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_EQ(r, 0x000F000F000F000Full);
}

/* ANDI/ORI/XORI zero-extend their immediate — unlike ADDIU, which sign-extends
 * it. Getting these two rules the wrong way round is a common emulator bug. */
static void t_immediate_logic_zero_extends(void)
{
    u64 r;
    u64 a = OPAQUE(0xFFFFFFFFFFFFFFFFull);

    __asm__ __volatile__(".set push; .set mips3\n\tandi %0, %1, 0xFFFF\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0x000000000000FFFFull);

    a = OPAQUE(0ull);
    __asm__ __volatile__(".set push; .set mips3\n\tori %0, %1, 0xFFFF\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0x000000000000FFFFull);

    __asm__ __volatile__(".set push; .set mips3\n\txori %0, %1, 0x8000\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0x0000000000008000ull);
}

static void t_addiu_immediate_sign_extends(void)
{
    u64 r;
    u64 a = OPAQUE(0ull);
    /* -1 as a 16-bit immediate, sign-extended to 32, then the 32-bit result
     * sign-extended to 64. */
    __asm__ __volatile__(".set push; .set mips3\n\taddiu %0, %1, -1\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0xFFFFFFFFFFFFFFFFull);
}

static void t_daddiu_immediate_sign_extends(void)
{
    u64 r;
    u64 a = OPAQUE(0ull);
    __asm__ __volatile__(".set push; .set mips3\n\tdaddiu %0, %1, -1\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0xFFFFFFFFFFFFFFFFull);
}

/* LUI's result is a 32-bit value, so bit 31 set means a sign-extended
 * register. */
static void t_lui_sign_extends(void)
{
    u64 r;
    __asm__ __volatile__(".set push; .set mips3\n\tlui %0, 0x8000\n\t.set pop"
                         : "=r"(r));
    CHECK_EQ(r, 0xFFFFFFFF80000000ull);

    __asm__ __volatile__(".set push; .set mips3\n\tlui %0, 0x7FFF\n\t.set pop"
                         : "=r"(r));
    CHECK_EQ(r, 0x000000007FFF0000ull);
}

/* ── set-less-than ────────────────────────────────────────────────────────── */

static void t_slt_is_signed_64bit(void)
{
    u64 r;
    u64 a = OPAQUE(0xFFFFFFFFFFFFFFFFull);   /* -1 */
    u64 b = OPAQUE(1ull);
    __asm__ __volatile__(".set push; .set mips3\n\tslt %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_EQ(r, 1ull);
    __asm__ __volatile__(".set push; .set mips3\n\tslt %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(b), "r"(a));
    CHECK_EQ(r, 0ull);
}

static void t_sltu_is_unsigned_64bit(void)
{
    u64 r;
    u64 a = OPAQUE(0xFFFFFFFFFFFFFFFFull);
    u64 b = OPAQUE(1ull);
    __asm__ __volatile__(".set push; .set mips3\n\tsltu %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(a), "r"(b));
    CHECK_EQ(r, 0ull);               /* 0xffff... is huge, not -1 */
    __asm__ __volatile__(".set push; .set mips3\n\tsltu %0, %1, %2\n\t.set pop"
                         : "=r"(r) : "r"(b), "r"(a));
    CHECK_EQ(r, 1ull);
}

/* SLTIU sign-extends its immediate, then compares unsigned — so `sltiu x, -1`
 * compares against 0xffffffffffffffff, and almost everything is less than it. */
static void t_sltiu_sign_extends_then_compares_unsigned(void)
{
    u64 r;
    u64 a = OPAQUE(1ull);
    __asm__ __volatile__(".set push; .set mips3\n\tsltiu %0, %1, -1\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 1ull);

    a = OPAQUE(0xFFFFFFFFFFFFFFFFull);
    __asm__ __volatile__(".set push; .set mips3\n\tsltiu %0, %1, -1\n\t.set pop"
                         : "=r"(r) : "r"(a));
    CHECK_EQ(r, 0ull);               /* equal, not less */
}

/* ── $zero ────────────────────────────────────────────────────────────────── */

static void t_zero_register_stays_zero(void)
{
    u64 r;
    __asm__ __volatile__(".set push; .set mips3; .set noat\n\t"
                         "daddiu $zero, $zero, 1234\n\t"
                         "move %0, $zero\n\t"
                         ".set pop" : "=r"(r));
    CHECK_EQ(r, 0ull);
}

static const struct test tests[] = {
    TEST("alu/addu_sign_extends",       t_addu_sign_extends,                   CPU_ALL),
    TEST("alu/addiu_sign_extends",      t_addiu_sign_extends,                  CPU_ALL),
    TEST("alu/subu_sign_extends",       t_subu_sign_extends,                   CPU_ALL),
    TEST("alu/sll_sign_extends",        t_sll_sign_extends,                    CPU_ALL),
    TEST("alu/srl_uses_low32",          t_srl_uses_low32,                      CPU_ALL),
    TEST("alu/sra_uses_low32",          t_sra_uses_low32,                      CPU_ALL),
    TEST("alu/variable_shift_mask",     t_variable_shift_masks_to_5_bits,      CPU_ALL),
    TEST("alu/dsll_dsrl_dsra",          t_dsll_dsrl_dsra,                      CPU_ALL),
    TEST("alu/dshift32_variants",       t_dshift32_variants,                   CPU_ALL),
    TEST("alu/dsllv_mask",              t_dsllv_masks_to_6_bits,               CPU_ALL),
    TEST("alu/add_overflow_traps",      t_add_overflow_traps,                  CPU_ALL),
    TEST("alu/addu_no_trap",            t_addu_does_not_trap,                  CPU_ALL),
    TEST("alu/addi_overflow_traps",     t_addi_overflow_traps,                 CPU_ALL),
    TEST("alu/add_overflow_negative",   t_add_overflow_negative_side,          CPU_ALL),
    TEST("alu/add_mixed_signs_ok",      t_add_no_overflow_on_mixed_signs,      CPU_ALL),
    TEST("alu/sub_overflow_traps",      t_sub_overflow_traps,                  CPU_ALL),
    TEST("alu/dadd_overflow_traps",     t_dadd_overflow_traps,                 CPU_ALL),
    TEST("alu/dadd_no_ov_at_32bit",     t_dadd_no_overflow_at_32bit_boundary,  CPU_ALL),
    TEST("alu/daddu_no_trap",           t_daddu_does_not_trap,                 CPU_ALL),
    TEST("alu/dsub_overflow_traps",     t_dsub_overflow_traps,                 CPU_ALL),
    TEST("alu/logical_64bit",           t_logical_ops_are_64bit,               CPU_ALL),
    TEST("alu/imm_logic_zero_extends",  t_immediate_logic_zero_extends,        CPU_ALL),
    TEST("alu/addiu_imm_sign_extends",  t_addiu_immediate_sign_extends,        CPU_ALL),
    TEST("alu/daddiu_imm_sign_extends", t_daddiu_immediate_sign_extends,       CPU_ALL),
    TEST("alu/lui_sign_extends",        t_lui_sign_extends,                    CPU_ALL),
    TEST("alu/slt_signed",              t_slt_is_signed_64bit,                 CPU_ALL),
    TEST("alu/sltu_unsigned",           t_sltu_is_unsigned_64bit,              CPU_ALL),
    TEST("alu/sltiu_sign_extend",       t_sltiu_sign_extends_then_compares_unsigned, CPU_ALL),
    TEST("alu/zero_register",           t_zero_register_stays_zero,            CPU_ALL),
};

const struct test_group group_alu = {
    "alu", tests, sizeof(tests) / sizeof(tests[0])
};
