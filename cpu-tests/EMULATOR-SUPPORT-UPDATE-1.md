# Update 1 to the emulator-support spec

Delta only — paste into the session already working from
`cpu-tests/EMULATOR-SUPPORT-PROMPT.md`. Everything not mentioned here is
unchanged.

---

**Update to Feature A (volume-directory writer / `mkvh`). New measured data —
Features B, C and D are unchanged, no action needed on those.**

I parsed a real SGI distribution CD to validate the volume-header layout
instead of leaving it to be discovered. Reference image:

```
~/IRIX 6.5.22 Installation Tools and Overlays (1 of 3).iso
```

`mkvh --dump` on that file must reproduce the following. These values were read
out of the image, so treat any disagreement as a bug in your parser:

```
magic 0x0BE5A941   rootpt 0  swappt 0  bootfile ''      csum VALID

volume directory @ 0x048            lbn = 512-byte block number
  [0] sgilabel   lbn=32      nbytes=512
  [1] mr         lbn=33      nbytes=19200000        (the miniroot)
  [2] sash64     lbn=37533   nbytes=266000
  [3] sashARCS   lbn=38053   nbytes=342532          (what an Indy boots)

partition table @ 0x138
  [ 7] nblks=1283016  first=48736  type=5  (SYSV)
  [ 8] nblks=48736    first=0      type=0  (VOLHDR)
  [10] nblks=1331776  first=0      type=6  (VOLUME)
```

**Two structural facts the writer must honour. Both differ from what
`src/sgi_vh.rs` does today, so they are behaviour changes, not just checks:**

1. **Partition 8 must span every volume-header file.** On the real CD it is
   48736 blocks (~24 MB) starting at 0, covering all the data the voldir points
   at; partition 7 begins immediately after it at block 48736. The existing
   scratch-image path hardcodes `VH_SECTORS = 8`, which is correct only for a
   volume header containing no files and wrong as soon as one is added. Sizing
   partition 8 to fit the voldir payload has to be part of the image builder.

2. **The filesystem partition is type 5 (`PT_SYSV`), not type 7 (`PT_EFS`)** —
   even though the filesystem living on it is EFS. Verified directly: the EFS
   superblock at block 1 of partition 7 has `fs_magic` = `0x00072959`
   (`EFS_MAGIC`) at offset `0x1c`, with `fs_size` = 1280320 sectors,
   `fs_cgfsize` = 8000, `fs_ncg` = 160. Emit what the shipping CD emits rather
   than what the type name suggests.

**Loader-format context — no action required, just so nothing surprises you
when you dump those files.** `sashARCS` is **ECOFF** (`f_magic` 0x0163 =
MIPSEBMAGIC, `a_magic` OMAGIC, `F_EXEC`), text at `0x10000000`, entry
`0x10021020`. `sash64` is an **ELF64 MSB `ET_REL`** object — a relocatable the
PROM relocates, which is what the PROM string `"Cannot load and relocate %s."`
refers to. So both PROM loader paths are exercised by real shipping media, and
`0x10000000` is a load address this PROM demonstrably accepts.

For Feature B, one related note that *reduces* scope: the test binary will
relocate itself into KSEG0 in its own startup code before running anything
(it has to — a suite that rewrites all 48 TLB entries cannot execute from a
TLB-mapped region like `0x10000000`). So `loadelf` only ever needs to honour
`p_vaddr` as given and set PC to `e_entry`; it does not need to set up any TLB
mappings or care which segment the address lands in.
