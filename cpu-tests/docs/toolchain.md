# Toolchain

## What the suite is built with

The pinned compiler (see below), targeting:

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

Cross packages may ship only an **o32** `libgcc.a` (Debian's does). Linking it into an
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

## Getting it

Every MIPS guest binary in the repository (this suite, `bench/`, and the
benchmark image iris embeds) is built with one compiler: GCC 16.2.1 for
`mips64-linux-gnu`, from Fedora 45. The compiler decides how much guest code a
kernel runs, so a different one is a different benchmark: GCC 14.2 runs
Dhrystone in 428 guest instructions per iteration where 16.2.1 needs 313.

The compiler runs in a container, so the build is the same on Linux, macOS and
Windows. Use Docker (Docker Desktop on macOS and Windows) or Podman, and run
the commands from the repository root. On Windows run them in a WSL 2 shell;
that path is untested.

Build the image once:

```sh
docker build -t iris-guest-toolchain -f cpu-tests/toolchain.Containerfile cpu-tests
```

Then run `make` in it:

```sh
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/src -w /src \
    iris-guest-toolchain make -C cpu-tests all
```

With Podman:

```sh
podman build -t iris-guest-toolchain -f cpu-tests/toolchain.Containerfile cpu-tests
podman run --rm --userns=keep-id --security-opt label=disable -v "$PWD":/src -w /src \
    iris-guest-toolchain make -C cpu-tests all
```

The targets you'll want:

| Command | Builds |
|---|---|
| `make -C cpu-tests all` | the cpu-tests guest |
| `make -C bench all` | the benchmark guest |
| `make -C bench prebuilt` | `bench/prebuilt/irisbench.elf`, from clean; commit it |

Mount the repository at `/src`: the build records paths relative to the
repository root, so the output doesn't depend on where your checkout is. Use
your machine's native architecture; emulating amd64 on arm64 crashed `cc1` at
random. Podman on Linux x86-64 and Docker on macOS arm64 produce identical bytes.

Installing Fedora 45's `gcc-mips64-linux-gnu` natively also works, and plain
`make` finds it. Any other MIPS cross compiler still builds for quick local
runs (`make CROSS=<prefix>`) with a warning; its binaries are not the ones CI
tests and its benchmark figures are not comparable.
