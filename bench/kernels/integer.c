/*
 * integer.c — integer CPU kernels.
 *
 * The first two are a deliberate pair. `alu` is one dependency chain, so every
 * operation waits for the one before it and the score is a latency; `alu_ilp`
 * is eight independent chains over the same operation mix, so the score is a
 * throughput. On real silicon the ratio between them is the machine's
 * superscalar width. Under an emulator it is something more useful: an
 * interpreter dispatches one instruction at a time whatever the dependencies,
 * so the two scores converge, while a JIT that has scheduled the block apart
 * pulls them back open. The gap is a direct read on how much real work the
 * translation layer is doing.
 */

#include "benchlib.h"

/* ── int/alu — dependent 32-bit ALU chain ─────────────────────────────────── */

#define ALU_OPS_PER_ROUND 16

static u32 alu_chain(u32 rounds, u32 seed)
{
    u32 x = seed, i;
    for (i = 0; i < rounds; i++) {
        x += 0x9E3779B9u;   x ^= x >> 15;
        x *= 0x85EBCA6Bu;   x ^= x >> 13;
        x += x << 3;        x ^= x >> 7;
        x |= 0x00000001u;   x -= 0x27D4EB2Fu;
        x ^= x << 5;        x *= 0xC2B2AE35u;
        x ^= x >> 16;       x += 0x165667B1u;
        x = (x << 11) | (x >> 21);
        x &= 0xFFFFFFFEu;   x |= 0x00000003u;
        x ^= 0x5BF03635u;
    }
    return x;
}

static u64 v_alu(void) { return cksum_u64(CKSUM_INIT, alu_chain(4096, 0x12345678u)); }
static u64 r_alu(u32 n) { SINK(alu_chain(n, 0x12345678u)); return (u64)n * ALU_OPS_PER_ROUND; }

/* ── int/alu_ilp — the same mix with eight independent chains ─────────────── */

static u32 alu_ilp(u32 rounds, u32 seed)
{
    u32 a = seed, b = seed ^ 0x11111111u, c = seed ^ 0x22222222u, d = seed ^ 0x33333333u;
    u32 e = seed ^ 0x44444444u, f = seed ^ 0x55555555u, g = seed ^ 0x66666666u, h = seed ^ 0x77777777u;
    u32 i;
    for (i = 0; i < rounds; i++) {
        a += 0x9E3779B9u; b += 0x9E3779B9u; c += 0x9E3779B9u; d += 0x9E3779B9u;
        e += 0x9E3779B9u; f += 0x9E3779B9u; g += 0x9E3779B9u; h += 0x9E3779B9u;
        a ^= a >> 15; b ^= b >> 15; c ^= c >> 15; d ^= d >> 15;
        e ^= e >> 15; f ^= f >> 15; g ^= g >> 15; h ^= h >> 15;
        a *= 0x85EBCA6Bu; b *= 0x85EBCA6Bu; c *= 0x85EBCA6Bu; d *= 0x85EBCA6Bu;
        e *= 0x85EBCA6Bu; f *= 0x85EBCA6Bu; g *= 0x85EBCA6Bu; h *= 0x85EBCA6Bu;
        a ^= a >> 13; b ^= b >> 13; c ^= c >> 13; d ^= d >> 13;
        e ^= e >> 13; f ^= f >> 13; g ^= g >> 13; h ^= h >> 13;
    }
    return a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
}

static u64 v_alu_ilp(void) { return cksum_u64(CKSUM_INIT, alu_ilp(4096, 0x12345678u)); }
static u64 r_alu_ilp(u32 n) { SINK(alu_ilp(n, 0x12345678u)); return (u64)n * 32; }

/* ── int/alu64 — 64-bit ALU, the half of MIPS III a 32-bit guest never uses ─ */

static u64 alu64_chain(u32 rounds, u64 seed)
{
    u64 x = seed;
    u32 i;
    for (i = 0; i < rounds; i++) {
        x += 0x9E3779B97F4A7C15ull;  x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ull;  x ^= x >> 27;
        x *= 0x94D049BB133111EBull;  x ^= x >> 31;
        x = (x << 17) | (x >> 47);
        x -= 0xD6E8FEB86659FD93ull;
        x &= 0xFFFFFFFFFFFFFFFEull;
        x |= 0x0000000000000003ull;
    }
    return x;
}

static u64 v_alu64(void) { return cksum_u64(CKSUM_INIT, alu64_chain(4096, 0x0123456789ABCDEFull)); }
static u64 r_alu64(u32 n) { SINK(alu64_chain(n, 0x0123456789ABCDEFull)); return (u64)n * 13; }

/* ── int/muldiv — mult/div, the long-latency integer units ────────────────── */

static u64 muldiv_chain(u32 rounds, u32 seed)
{
    u32 a = seed | 1u, i;
    u64 acc = 0;
    for (i = 0; i < rounds; i++) {
        u32 b = a * 2654435761u + 1u;
        u32 q, r;
        if (b == 0) b = 1;
        q = a / b;
        r = a % b;
        acc += (u64)q * 31u + r;
        {
            /* 64-bit divide too: dmultu/ddivu are a different unit again, and
             * an emulator that special-cases the 32-bit case will show it. */
            u64 wide = ((u64)a << 20) ^ 0x5DEECE66Dull;
            u64 den = (u64)b | 1ull;
            acc ^= wide / den;
            acc += wide % den;
        }
        a = a * 1103515245u + 12345u;
        a |= 1u;
    }
    return acc ^ a;
}

static u64 v_muldiv(void) { return cksum_u64(CKSUM_INIT, muldiv_chain(2048, 0xC0FFEE11u)); }
static u64 r_muldiv(u32 n) { SINK(muldiv_chain(n, 0xC0FFEE11u)); return (u64)n * 6; }

/* ── int/branch — data-dependent, unpredictable branches ──────────────────── */

/* The branch direction comes out of a hash, so no predictor helps and no
 * translator can turn it into straight-line code. This is the shape that hurts
 * an emulator most: a taken branch is a dispatch, and a region-compiling JIT
 * has to leave its compiled region at each one it cannot prove. */
static u32 branch_maze(u32 rounds, u32 seed)
{
    u32 x = seed, acc = 0, i;
    for (i = 0; i < rounds; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        if (x & 1u)          acc += 3;
        else                 acc ^= 0x1234u;
        if ((x & 6u) == 4u)  acc -= 7;
        else if (x & 8u)     acc += acc >> 3;
        else                 acc ^= x;
        switch ((x >> 4) & 7u) {
        case 0: acc += 11; break;
        case 1: acc -= 13; break;
        case 2: acc ^= 17; break;
        case 3: acc += acc << 2; break;
        case 4: acc = ~acc; break;
        case 5: acc ^= x >> 8; break;
        case 6: acc += 19; break;
        default: acc -= 23; break;
        }
        if ((s32)acc < 0) acc = (u32)(-(s32)acc);
    }
    return acc ^ x;
}

static u64 v_branch(void) { return cksum_u64(CKSUM_INIT, branch_maze(8192, 0xDEADBEEFu)); }
static u64 r_branch(u32 n) { SINK(branch_maze(n, 0xDEADBEEFu)); return (u64)n * 4; }

/* ── int/bitops — bit manipulation, no multiply, no memory ────────────────── */

static u32 popcnt32(u32 v)
{
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;
    return (v * 0x01010101u) >> 24;
}

static u32 bitops_chain(u32 rounds, u32 seed)
{
    u32 x = seed, acc = 0, i;
    for (i = 0; i < rounds; i++) {
        u32 rev = 0, k;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        for (k = 0; k < 32; k++) rev = (rev << 1) | ((x >> k) & 1u);
        acc += popcnt32(x) + popcnt32(rev);
        acc ^= rev;
        acc = (acc << 7) | (acc >> 25);
    }
    return acc;
}

static u64 v_bitops(void) { return cksum_u64(CKSUM_INIT, bitops_chain(1024, 0x0BADC0DEu)); }
static u64 r_bitops(u32 n) { SINK(bitops_chain(n, 0x0BADC0DEu)); return (u64)n * 80; }

/* ══ int/dhrystone — Dhrystone 2.1 ════════════════════════════════════════════
 *
 * Weicker's benchmark, in the shape the 1988 C version fixed: same procedure
 * and function decomposition, same string and record traffic, same globals. It
 * is here because DMIPS (runs per second / 1757) is a number that can be
 * compared against forty years of published figures for real hardware,
 * including SGI's own, which is exactly what "how fast is the emulated Indy"
 * needs and what a bespoke kernel can never provide.
 *
 * Two departures, both forced by running with no operating system, and neither
 * touching the measured work: strcpy/strcmp are the local d_* versions below
 * (there is no libc), and the timing loop belongs to the harness rather than
 * to Proc_0 (there is no clock() and the harness has a better one).
 */

typedef enum { Ident_1, Ident_2, Ident_3, Ident_4, Ident_5 } Enumeration;
typedef int  One_Thirty;
typedef int  One_Fifty;
typedef char Capital_Letter;
typedef int  Boolean;
typedef char Str_30[31];
typedef int  Arr_1_Dim[50];
typedef int  Arr_2_Dim[50][50];

typedef struct record {
    struct record *Ptr_Comp;
    Enumeration    Discr;
    union {
        struct { Enumeration Enum_Comp; One_Thirty Int_Comp; char Str_Comp[31]; } var_1;
        struct { Enumeration E_Comp_2;  char Str_2_Comp[31]; } var_2;
        struct { char Ch_1_Comp; char Ch_2_Comp; } var_3;
    } variant;
} Rec_Type, *Rec_Pointer;

static Rec_Pointer  Ptr_Glob, Next_Ptr_Glob;
static int          Int_Glob;
static Boolean      Bool_Glob;
static char         Ch_1_Glob, Ch_2_Glob;
static int          Arr_1_Glob[50];
static int          Arr_2_Glob[50][50];
static Rec_Type     Rec_Glob_1, Rec_Glob_2;

static void d_strcpy(char *d, const char *s) { while ((*d++ = *s++) != 0) { } }
static int  d_strcmp(const char *a, const char *b)
{
    while (*a && *a == *b) { a++; b++; }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

static Boolean Func_3(Enumeration Enum_Par_Val)
{
    Enumeration Enum_Loc = Enum_Par_Val;
    return Enum_Loc == Ident_3;
}

static void Proc_7(One_Fifty Int_1_Par_Val, One_Fifty Int_2_Par_Val, One_Fifty *Int_Par_Ref)
{
    One_Fifty Int_Loc = Int_1_Par_Val + 2;
    *Int_Par_Ref = Int_2_Par_Val + Int_Loc;
}

static void Proc_6(Enumeration Enum_Val_Par, Enumeration *Enum_Ref_Par)
{
    *Enum_Ref_Par = Enum_Val_Par;
    if (!Func_3(Enum_Val_Par)) *Enum_Ref_Par = Ident_4;
    switch (Enum_Val_Par) {
    case Ident_1: *Enum_Ref_Par = Ident_1; break;
    case Ident_2: *Enum_Ref_Par = (Int_Glob > 100) ? Ident_1 : Ident_4; break;
    case Ident_3: *Enum_Ref_Par = Ident_2; break;
    case Ident_4: break;
    case Ident_5: *Enum_Ref_Par = Ident_3; break;
    }
}

static void Proc_8(Arr_1_Dim Arr_1_Par_Ref, Arr_2_Dim Arr_2_Par_Ref,
                   int Int_1_Par_Val, int Int_2_Par_Val)
{
    One_Fifty Int_Index, Int_Loc = Int_1_Par_Val + 5;
    Arr_1_Par_Ref[Int_Loc] = Int_2_Par_Val;
    Arr_1_Par_Ref[Int_Loc + 1] = Arr_1_Par_Ref[Int_Loc];
    Arr_1_Par_Ref[Int_Loc + 30] = Int_Loc;
    for (Int_Index = Int_Loc; Int_Index <= Int_Loc + 1; ++Int_Index)
        Arr_2_Par_Ref[Int_Loc][Int_Index] = Int_Loc;
    Arr_2_Par_Ref[Int_Loc][Int_Loc - 1] += 1;
    Arr_2_Par_Ref[Int_Loc + 20][Int_Loc] = Arr_1_Par_Ref[Int_Loc];
    Int_Glob = 5;
}

static Enumeration Func_1(Capital_Letter Ch_1_Par_Val, Capital_Letter Ch_2_Par_Val)
{
    Capital_Letter Ch_1_Loc = Ch_1_Par_Val;
    Capital_Letter Ch_2_Loc = Ch_1_Loc;
    if (Ch_2_Loc != Ch_2_Par_Val) return Ident_1;
    Ch_1_Glob = Ch_1_Loc;
    return Ident_2;
}

static Boolean Func_2(Str_30 Str_1_Par_Ref, Str_30 Str_2_Par_Ref)
{
    One_Thirty Int_Loc = 2;
    Capital_Letter Ch_Loc = 'A';
    while (Int_Loc <= 2) {
        if (Func_1(Str_1_Par_Ref[Int_Loc], Str_2_Par_Ref[Int_Loc + 1]) == Ident_1) {
            Ch_Loc = 'A';
            Int_Loc += 1;
        }
    }
    if (Ch_Loc >= 'W' && Ch_Loc < 'Z') Int_Loc = 7;
    if (Ch_Loc == 'R') return 1;
    if (d_strcmp(Str_1_Par_Ref, Str_2_Par_Ref) > 0) { Int_Loc += 7; Int_Glob = Int_Loc; return 1; }
    return 0;
}

static void Proc_3(Rec_Pointer *Ptr_Ref_Par)
{
    if (Ptr_Glob != 0) *Ptr_Ref_Par = Ptr_Glob->Ptr_Comp;
    Proc_7(10, Int_Glob, &Ptr_Glob->variant.var_1.Int_Comp);
}

static void Proc_1(Rec_Pointer Ptr_Val_Par)
{
    Rec_Pointer Next_Record = Ptr_Val_Par->Ptr_Comp;
    /* structassign in the original — a whole-record copy, which is exactly the
     * memcpy the compiler emits here. */
    *Ptr_Val_Par->Ptr_Comp = *Ptr_Glob;
    Ptr_Val_Par->variant.var_1.Int_Comp = 5;
    Next_Record->variant.var_1.Int_Comp = Ptr_Val_Par->variant.var_1.Int_Comp;
    Next_Record->Ptr_Comp = Ptr_Val_Par->Ptr_Comp;
    Proc_3(&Next_Record->Ptr_Comp);
    if (Next_Record->Discr == Ident_1) {
        Next_Record->variant.var_1.Int_Comp = 6;
        Proc_6(Ptr_Val_Par->variant.var_1.Enum_Comp, &Next_Record->variant.var_1.Enum_Comp);
        Next_Record->Ptr_Comp = Ptr_Glob->Ptr_Comp;
        Proc_7(Next_Record->variant.var_1.Int_Comp, 10, &Next_Record->variant.var_1.Int_Comp);
    } else {
        *Ptr_Val_Par = *Ptr_Val_Par->Ptr_Comp;
    }
}

static void Proc_2(One_Fifty *Int_Par_Ref)
{
    One_Fifty Int_Loc = *Int_Par_Ref + 10;
    Enumeration Enum_Loc = Ident_1;   /* the original leaves this uninitialised;
                                       * seeding it keeps the checksum stable
                                       * without changing any executed path,
                                       * since Int_Glob is 5 here every time */
    for (;;) {
        if (Ch_1_Glob == 'A') {
            Int_Loc -= 1;
            *Int_Par_Ref = Int_Loc - Int_Glob;
            Enum_Loc = Ident_1;
        }
        if (Enum_Loc == Ident_1) break;
    }
}

static void Proc_4(void)
{
    Boolean Bool_Loc = Ch_1_Glob == 'A';
    Bool_Glob = Bool_Loc | Bool_Glob;
    Ch_2_Glob = 'B';
}

static void Proc_5(void)
{
    Ch_1_Glob = 'A';
    Bool_Glob = 0;
}

static void dhry_setup(void)
{
    Next_Ptr_Glob = &Rec_Glob_1;
    Ptr_Glob      = &Rec_Glob_2;

    Ptr_Glob->Ptr_Comp                    = Next_Ptr_Glob;
    Ptr_Glob->Discr                       = Ident_1;
    Ptr_Glob->variant.var_1.Enum_Comp     = Ident_3;
    Ptr_Glob->variant.var_1.Int_Comp      = 40;
    d_strcpy(Ptr_Glob->variant.var_1.Str_Comp, "DHRYSTONE PROGRAM, SOME STRING");

    Next_Ptr_Glob->Ptr_Comp               = 0;
    Next_Ptr_Glob->Discr                  = Ident_1;
    Next_Ptr_Glob->variant.var_1.Enum_Comp = Ident_3;
    Next_Ptr_Glob->variant.var_1.Int_Comp  = 0;
    Next_Ptr_Glob->variant.var_1.Str_Comp[0] = 0;

    Arr_2_Glob[8][7] = 10;
    Int_Glob = 0;
    Bool_Glob = 0;
    Ch_1_Glob = 0;
    Ch_2_Glob = 0;
    {
        int i, j;
        for (i = 0; i < 50; i++) { Arr_1_Glob[i] = 0; for (j = 0; j < 50; j++) Arr_2_Glob[i][j] = 0; }
        Arr_2_Glob[8][7] = 10;
    }
}

static void dhry_runs(u32 runs)
{
    One_Fifty      Int_1_Loc, Int_2_Loc, Int_3_Loc;
    char           Ch_Index;
    Enumeration    Enum_Loc;
    Str_30         Str_1_Loc, Str_2_Loc;
    u32            Run_Index;

    d_strcpy(Str_1_Loc, "DHRYSTONE PROGRAM, 1'ST STRING");
    d_strcpy(Str_2_Loc, "DHRYSTONE PROGRAM, 2'ND STRING");
    /* All three are assigned on every iteration of the loop below; seeding
     * them keeps a runs==0 call (which the autoscaler never makes, but the
     * compiler cannot know that) from reading an uninitialised local. */
    Int_1_Loc = 0;
    Int_2_Loc = 0;
    Int_3_Loc = 0;
    Enum_Loc  = Ident_1;

    for (Run_Index = 1; Run_Index <= runs; ++Run_Index) {
        Proc_5();
        Proc_4();
        Int_1_Loc = 2;
        Int_2_Loc = 3;
        d_strcpy(Str_2_Loc, "DHRYSTONE PROGRAM, 2'ND STRING");
        Enum_Loc = Ident_2;
        Bool_Glob = !Func_2(Str_1_Loc, Str_2_Loc);
        while (Int_1_Loc < Int_2_Loc) {
            Int_3_Loc = 5 * Int_1_Loc - Int_2_Loc;
            Proc_7(Int_1_Loc, Int_2_Loc, &Int_3_Loc);
            Int_1_Loc += 1;
        }
        Proc_8(Arr_1_Glob, Arr_2_Glob, Int_1_Loc, Int_3_Loc);
        Proc_1(Ptr_Glob);
        for (Ch_Index = 'A'; Ch_Index <= Ch_2_Glob; ++Ch_Index) {
            if (Enum_Loc == Func_1(Ch_Index, 'C')) {
                Proc_6(Ident_1, &Enum_Loc);
                d_strcpy(Str_2_Loc, "DHRYSTONE PROGRAM, 3'RD STRING");
                Int_2_Loc = (int)Run_Index;
                Int_Glob  = (int)Run_Index;
            }
        }
        Int_2_Loc = Int_2_Loc * Int_1_Loc;
        Int_1_Loc = Int_2_Loc / Int_3_Loc;
        Int_2_Loc = 7 * (Int_2_Loc - Int_3_Loc) - Int_1_Loc;
        Proc_2(&Int_1_Loc);
    }

    /* Feed the loop-carried locals somewhere the optimiser cannot see through,
     * so the last iteration is not dead code. */
    SINK(Int_1_Loc); SINK(Int_2_Loc); SINK(Int_3_Loc);
    SINK((int)Enum_Loc); SINK(Str_2_Loc[0]);
}

static u64 v_dhry(void)
{
    u64 h = CKSUM_INIT;
    int i, j;
    dhry_setup();
    dhry_runs(1000);
    h = cksum_u64(h, (u64)(u32)Int_Glob);
    h = cksum_u64(h, (u64)(u32)Bool_Glob);
    h = cksum_u64(h, (u64)(u8)Ch_1_Glob);
    h = cksum_u64(h, (u64)(u8)Ch_2_Glob);
    for (i = 0; i < 50; i++) h = cksum_u64(h, (u64)(u32)Arr_1_Glob[i]);
    for (i = 0; i < 50; i++) for (j = 0; j < 50; j++) h = cksum_u64(h, (u64)(u32)Arr_2_Glob[i][j]);
    h = cksum_u64(h, (u64)(u32)Ptr_Glob->variant.var_1.Int_Comp);
    h = cksum_u64(h, (u64)(u32)Next_Ptr_Glob->variant.var_1.Int_Comp);
    h = cksum_bytes(h, Ptr_Glob->variant.var_1.Str_Comp, 31);
    return h;
}

static u64 r_dhry(u32 n) { dhry_setup(); dhry_runs(n); return (u64)n; }

/* ── registration ─────────────────────────────────────────────────────────── */

static const struct bench benches[] = {
    BENCH("int/alu",       "ops",  v_alu,     r_alu,     1u << 14, BG_INT),
    BENCH("int/alu_ilp",   "ops",  v_alu_ilp, r_alu_ilp, 1u << 14, BG_INT),
    BENCH("int/alu64",     "ops",  v_alu64,   r_alu64,   1u << 14, BG_INT),
    BENCH("int/muldiv",    "ops",  v_muldiv,  r_muldiv,  1u << 13, BG_INT),
    BENCH("int/branch",    "ops",  v_branch,  r_branch,  1u << 13, BG_INT),
    BENCH("int/bitops",    "ops",  v_bitops,  r_bitops,  1u << 10, BG_INT),
    BENCH("int/dhrystone", "dhry", v_dhry,    r_dhry,    1u << 12, BG_INT),
};

const struct bench_group group_integer = {
    "integer", benches, sizeof(benches) / sizeof(benches[0])
};
