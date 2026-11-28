# fx label/create/all writes root and swap partitions, not just a label

Status: measured 2026-11-22 (Indy, embedded PROM, `fx.ARCS` from the IRIX
6.5.22 Installation Tools CD, "SGI Version 6.5 ARCS BE Oct 6, 2003").

Question: does `fx` only label a new disk, with the swap partition made later
by the miniroot? No. `fx` lays out root and swap itself.

## What was run

A blank 2 GB raw image at SCSI 1 (`overlay = false`), headless, driven over
`iris-ci`:

```
5                                                  (Command Monitor)
boot -f dksc(0,4,8)sashARCS dksc(0,4,7)stand/fx.ARCS --x
<enter> <enter> <enter>                            (dksc, ctlr 0, drive 1)
l / c / a      label/create/all
.. / sy        sync
.. / exi
```

On opening the blank disk fx prints "volume header not valid", "creating
default bootinfo", "created default partitions, use /repartition menu to
change", "creating default volume directory". `label/show/partitions` already
lists the table below at that point, before anything is written: the image
was still all zeros. `label/create/all` repeats the same defaults and adds
sgiinfo; only `sync` ("writing label info") touches the disk.

## What is on the disk afterwards

`mkvh dump disk.img`:

```
rootpt 0   swappt 1   bootfile "/unix"   checksum valid
volume directory (0 of 15 entries)
slot  0  first  266240  nblks 3928064  type 10 (PT_XFS)     root, 1918 MB
slot  1  first    4096  nblks  262144  type  3 (PT_RAW)     swap, 128 MB
slot  8  first       0  nblks    4096  type  0 (PT_VOLHDR)  2 MB
slot 10  first       0  nblks 4194304  type  6 (PT_VOLUME)  whole disk
```

Exactly two sectors are non-zero: sector 0 (the volume header) and sector 2
(sgiinfo). No filesystem is written; the root is only typed XFS. Swap sits
straight after the volume header and root takes the rest. `sash` and `ide`
are not in the volume directory yet: the installer adds them.

## Consequence

A front-end that wanted to "write the label as fx does" would have to choose
the root and swap split too, which is the operator's decision in fx
(`/repartition`). So iris does not pre-provision disks: it creates an empty
image and the user runs fx, as `irix-install.md` describes.

Not measured: IRIX 5.3's fx, other disk sizes (the 128 MB swap may scale),
and option-drive layouts.
