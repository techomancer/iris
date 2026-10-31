# VPro (Odyssey) on the Octane: boards, versions and bring-up

Status: howto (iris `src/dev/odyssey/`, 2026-10-30). The board's contract is
origami's reviewed datasheet `docs/datasheets/odyssey.md`; this note is what
you need to start a guest and read a trace.

**What it is.** The Octane's second graphics option after IMPACT: a
crosstalk widget (part 0xc013, manufacturer 0x023) on widget 11, where the
IMPACT board otherwise sits. Everything goes through a command FIFO; there
is no linear frame buffer. `public-references.md` lists the public code for
the board.

**Selecting it.** `[graphics] board = "v6" | "v8" | "v10" | "v12"` (also
`vpro:v6` and so on). `impact` stays the Octane's default and the value is
refused on the other machines. `perfetto.md` in `docs/` lists the VPro
tracks.

## Board catalogue

| Board | Widget revision | gfxinfo BUZZ | Memory | Banks | Identity record |
|---|---|---|---|---|---|
| V6 | 2 | A.1 | 32 MB | 2 | `ODY32` (030-1403) |
| V8 | 2 | A.1 | 128 MB | 4 | `ODY128` (030-1404) |
| V10 | 10 | B.1 | 32 MB | 2 | `ODY32VERSIONB` (030-1655) |
| V12 | 10 | B.1 | 128 MB | 4 | `ODY128VERSIONB` (030-1611) |

PB&J version is 1 on all four. Commands, frame buffers and textures share
one SDRAM pool on every board. Inferred, not captured: the V8's A.1 and
banks, the V10's revision 10, and the 32 MB boards' two banks (the datasheet
marks them). The Fuel and Tezro boards are other assemblies; the Octane
names are answered on those machines too, a known limitation.

Guest-side consequences worth knowing before you debug a "wrong" screen:

- The X server refuses 1280x1024_60 at 16-byte depth on a BUZZ A.1 board.
  V6 and V8 therefore start in `1280x720_72`, V10 and V12 in
  `1280x1024_60`. That is the guest's decision, not a model bug.
- `GL_MAX_TEXTURE_SIZE` is 512 below 128 MB and 2048 otherwise.
- On a small board the kernel's board-memory manager pages objects to host
  memory when the pool fills.

## Crossbow and PROM minimums

SGI's VPro upgrade kit (007-4210-003) needs Crossbow 1.4, IRIX 6.5.10, and
PROM 4.5 for V10/V12 or PROM 4.0 for V6/V8. The Octane presents Crossbow
revision code 4. An Octane2's Crossbow reads code 5 at the PROM and shows as
1.4 in its inventory, so a machine with a V10 or V12 presents code 5.

Code 5 has registers the earlier parts lack: a reset fence for the core and
one per link at 0x78 to 0xb8 (reset 0xff, bits 7:0 kept), a resource lock at
0xc0 that takes a value only while free, and its clear at 0xc8. The
Octane2's PROM programs the fences; answering them with an access error, as
code 4 does, leaves it off its path. The fences gate nothing in the model,
because no warm reset travels through the modelled switch. Revision code 4
and earlier keep the range reserved.

iris warns at start-up when the PROM image's banner is older than the board
needs. `IP30prom.bin` (from IRIX 6.5.30) holds two PROMs: Rev 4.9 is the
recovery PROM and Rev 4.17 the main one; the PROM console reports 4.17, as
`flash -V` does in IRIX.

## Disk images

The VPro needs a kernel driver and X module (`odyssey.so`, `libGLcore` for
`IP30ODSY64`) that an IMPACT install does not carry; an IMPACT install boots,
then X finds no screen. In the scratch inputs, `octane2-vpro-*.raw` and
`vpro-octane2-installed-*.raw` are VPro installs (IRIX 6.5.30 on an Octane2
V12, which boots to the desktop with `glxinfo` reporting `VPRO/B/128`);
`si-octane-*.raw` is an IMPACT install. Names are the only marker; confirm
with `hinv` or `gfxinfo` after boot.

## Checking it came up

- `gfxinfo` prints `BUZZ version B.1`, `PB&J version 1`, memory size.
- `glxinfo` renderer is `VPRO/<A|B>/<MB>`.
- `hinv` names V6/V8/V10/V12 from generation and size.
- `GL_ODY_*` environment variables override libGLcore's fast-path defaults,
  which differ between BUZZ A and B.
