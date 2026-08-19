# Toolchain

## What the suite is built with

`mips-linux-gnu-gcc` (Debian/Ubuntu cross GCC), targeting:

```
-march=mips3 -mabi=n32 -EB -mno-abicalls -fno-pic -G0 -msoft-float
```

Two of those are load-bearing and were arrived at the hard way.

### `-mabi=n32`, not `-mabi=32`

Under **o32 a `long long` lives in an even/odd register pair**, so an inline-asm
operand like `"=r"(u64)` binds only the *first* register of the pair. Every
64-bit test silently reads half a result — and on a big-endian target the half
it reads is the high word, so `addu` producing `0x80000000` came back as
`0x8000000000000000` and looked like a CPU bug. n32 keeps 64-bit values in
single registers, which is exactly what a MIPS III/IV suite needs.

n32 is still **ELF32 MSB** (`elf32-ntradbigmips`), so `--load-elf` and the PROM
both take it; only `e_flags` differs (`abi2`).

### No libgcc

The Debian cross package ships a **single, o32** `libgcc.a`. Linking it into an
n32 image would be silently wrong. It is also unnecessary: under n32 every
64-bit operation the C code performs — including the `% 10` / `/= 10` in
`con_udec` — is a native instruction rather than a `__udivdi3` call. `LIBGCC`
is deliberately empty in the Makefile, so if a helper reference ever does
appear the link fails loudly instead of pulling in the wrong ABI.

### Numeric registers in `start.S`

o32 and n32/n64 disagree about which hardware register each `$tN` name means,
and under n32 `$t4`–`$t7` **do not exist** (`$8`–`$11` are `$a4`–`$a7` there).
The bootstrap uses numeric names (`$8`–`$14`) so it assembles identically under
any ABI.

## Installing it

With root:

```sh
sudo apt-get install gcc-mips-linux-gnu binutils-mips-linux-gnu
```

Without root — what an unattended session has to do — unpack the same `.deb`s
into a prefix. `toolchain.mk` looks there automatically:

```sh
make -C cpu-tests toolchain-local
```

That fetches `binutils-mips-linux-gnu`, `binutils-common`, `libbinutils`,
`gcc-12-mips-linux-gnu`, `cpp-12-mips-linux-gnu` and `libgcc-12-dev-mips-cross`
and `dpkg -x`s them into `~/.local/opt/mips-linux-gnu`.

One wrinkle: the binutils from the `.deb` link against their own
`libbfd-2.42-mips.so` / `libopcodes-2.42-mips.so`, which are not on the default
loader path in a rootless install. `toolchain.mk` exports `LD_LIBRARY_PATH` for
them; `env.sh` in the prefix does the same for an interactive shell.

Override the prefix with `make CROSS=mips64-elf-` for a different toolchain.
