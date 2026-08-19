# Emulator support for bare-metal CPU tests

Four additive, default-off features for running a self-checking bare-metal MIPS
binary under IRIS. None of them changes an ordinary IRIX boot.

## A — `mkvh`: bootable volume headers

`src/sgi_vh.rs` gained the volume directory (0x048, 15 × 16 bytes), so an image
can carry named standalone files the PROM boots by name.

```sh
mkvh build boot.img --size 16M sashARCS=./sashARCS      # bootable by name
mkvh build boot.img --size 64M --part 7:5:+:2048=efs.img cputest=./cputest
mkvh dump /path/to/sgi.iso                              # parse real media
```

`--part slot:type:first:nblks[=file]`; `first` may be `+`, meaning "start where
the volume header ends", which is what SGI's own media does. **An EFS filesystem
partition is type 5 (`PT_SYSV`)**, not 7 — see
`rules/testing/sgi-volume-directory-layout-verified-on-real-media.md`.

From the PROM: `boot -f dksc(0,<scsi-id>,8)<name>`.

## B — `--load-elf` / monitor `loadelf`

Load a static ELF32 MSB `ET_EXEC` for `EM_MIPS` straight into RAM and jump to
its entry point — no image to rebuild between edits.

```sh
iris --load-elf ./cputest.elf          # cold start, no PROM involved
```
```
> cpu stop
> loadelf ./cputest.elf                # segments + entry are printed
> loadbin ./blob.bin 0x88300000        # raw bytes, PC untouched
> run
```

Loading requires a stopped CPU — that is what keeps the JIT honest, since its
block cache is owned by the dispatch loop and dies with the CPU thread. The
loaded range is flushed out of L1D/L2 and invalidated out of L1I, so reloading a
different binary at the same address runs the new code.

Link at **0x88000000+** (Indy RAM starts at physical 0x08000000). Before POST no
RAM is mapped at all; `--load-elf` maps the banks the way POST does, and every
load probes the bus first and refuses rather than writing into the void. See
`rules/testing/loading-code-before-post-ram-is-unmapped.md`.

## C — TFTP for PROM network boot

```sh
iris --tftp-dir ./build     # or [network] tftp_dir = "./build"
```
```
>> boot -f bootp()cputest
```

Read-only RFC 1350 inside the NAT gateway (no host socket, no new entitlement).
RRQ only, octet only, 512-byte ACK-driven blocks, bounded retransmits, and paths
that escape the root — `..`, absolute, or via symlink — are refused. Unset means
no TFTP at all.

## D — test device (`--test-device`)

MMIO at `0x1F400000` (GIO expansion slot 0, empty on a stock Indy), 64 KB, four
word registers repeating every 16 bytes:

| Offset | Access | Behaviour |
|---|---|---|
| `0x00` | read | `SIGNATURE` = `0x49524953` (`"IRIS"`). Absent on real hardware, so the suite falls back to SCC-only output. |
| `0x04` | write | `PUTC` — low byte to the host's stdout. |
| `0x08` | write | `DUMP` — full machine state to JSON; the written value tags the dump point. |
| `0x0C` | write | `EXIT` — terminate the emulator with the written value as the process exit code. |

```sh
iris --test-device --test-device-dump ./dump.json --load-elf ./cputest.elf
```

The dump carries `gpr[32]`, `hi`/`lo`, `pc`, the CP0 registers by name,
`fpr[32]` plus FCSR/FIR, and identifies R4400 vs R5000 from PRId.

`--cheritest-dump-hook` additionally makes a guest write to **CP0 register 26**
trigger the same dump, the convention the external `cheritest` corpus uses. It
is off by default and must stay that way for a normal boot: CP0 26 is ECC on a
real R4400 and the PROM writes it during cache initialisation.

## Getting results back

Exit code for the verdict, stdout for the log, the dump for diagnosis:

| Channel | Carries | Host side |
|---|---|---|
| `EXIT` | pass/fail | process exit code — no string matching, works headless |
| `PUTC` | per-test lines | stdout |
| `DUMP` | state at a failure | JSON, diffable between runs |
| SCC ttyd1 | the same lines on real hardware | `--serial-log FILE` |

Prefer `--serial-log` over scraping TCP 8881: the serial backend holds one
client and only notices a dead one on a failed write
(`rules/testing/scc-serial-output-from-bare-metal-code.md`).
