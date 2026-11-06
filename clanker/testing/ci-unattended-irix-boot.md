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
→ quit. The next (graphics) run auto-boots the disk unattended. Leave
`console` unset (= `g`) if you want the graphical login on screen; `console=d`
puts everything on serial and leaves the framebuffer dark (irix-install.md §8).

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
- Detecting login with a graphics console needs a getty on ttyd1 in the image;
  otherwise serial is silent after the PROM.
