# Unattended IRIX boot under --ci: seed NVRAM, composite screenshots on the CPU

**Keywords:** ci,boot-test,nvram,setenv,screenshot,black,headless,console,SystemPartition
**Category:** testing

Two things stop a scripted `--ci` boot of an installed disk from working out of
the box. `scripts/ci-boot-test.sh` handles both; read it before writing another.
The `boot-test` job in `.github/workflows/build.yml` runs it.

## A fresh NVRAM can't boot anything

With no `nvram.bin` (or a bad checksum) the PROM prints `NVRAM checksum is
incorrect: reinitializing.` and resets to defaults that have **no
`SystemPartition`/`OSLoadPartition`**. AutoLoad still fires, and on a graphics
console it stops at `Cannot load /sash. No default device and path in
environment.` iris injects only the MAC (`backdoor_set_mac_if_blank`), never
the boot path.

Fix: a headless pass first. With `--headless` the PROM falls back to the serial
console, which the CI socket can drive: wait `Option?` → `5` → `setenv -f
SystemPartition scsi(0)disk(1)rdisk(0)partition(8)`, `OSLoadPartition
...partition(0)`, `OSLoadFilename /unix`, `AutoLoad Yes` → `iris-ci rtc-save`
→ quit. The next run auto-boots the disk unattended. The script also sets
`console d`, so the kernel and rc scripts print on serial.

## The disk needs a getty on ttyd1

`console` moves the PROM, kernel and rc output, not the login. `login:` on
serial comes from the getty that `/etc/inittab` runs on `ttyd1` (its `t1`
entry), and with the install's default `chkconfig` (windowsystem, xdm and
visuallogin on) IRIX 5.3 still starts X and draws the Visual Login when
`console=d`; the framebuffer is not dark. Measured 2026-11-10 with the
nightly CLI on a fresh install from `IRIX 5.3 for Indy including R5000.iso`:

- stock inittab: `IRIS login:` on serial 30 s into the boot, with `console d`
  and with `console` left at `g`.
- the same disk with only `t1` set to `off`: serial ends after the rc
  scripts' output, never shows `login:`, and a screenshot shows the Visual
  Login. The CI disk's nightly boots look the same: with `console d` serial
  ends after the rc output, with `g` after the PROM, and the screenshots the
  first version of the test took show its Visual Login within 75 s.

So a disk that fails here with a transcript ending in rc output most likely
has no getty on `ttyd1`; check its `/etc/inittab` before blaming the boot.

## Each check is a JUnit case

The script records each check (seeding NVRAM, the login prompt, iris quitting
when asked) with `check`, and its exit trap writes them all to
`boot-test/boot-test.xml` with `iris-ci junit`, also after a failure that stops
the script early. The job hands that file to the JUnit reporter with
`annotate_only`, which needs no `checks: write`, so a failed run is annotated
with the check that failed. Add a check the same way rather than another exit
code.

## Screenshots under --ci

`cmd_screenshot` (src/ci.rs) takes its frame from `gfx_display::fresh_frame`.
With no front-end, Newport composes in software on its refresh thread, so this
needs no GPU and works on headless CI runners.

The first screenshot right after `start` fails with `the display has no video
mode yet`: VC2 timings aren't programmed. Wait ~15s.

## Other gotchas hit on the way

- macOS caps a Unix socket path at 104 bytes (`path must be shorter than
  SUN_LEN`); keep `ci_socket` under `/tmp`, not a deep workspace path.
- A config with no `[scsi.1]` falls back to `scsi1.raw` and iris exits fatally
  if it's missing.
