---
name: irix-inst-packaging
description: >-
  Build, inspect, install and remove IRIX inst software packages (.tardist
  and dist directories) and SGI-style rollback patches for any software:
  stage files, write the idb and spec, run gendist, tag and strip binaries,
  add desktop integration (filetype rules, icons, toolchest, icon catalog)
  and install-time actions (terminfo, filetype database, rqs), test with
  inst -a in the IRIX guest, and unpack SGI packages on the host with uninst.
---

# IRIX inst packaging and patching

IRIX installs software with `inst` (and its GUI `swmgr`) from
*distributions*. This skill covers making them for anything: applications
(ttr, dash, chocolate-doom), library upgrades (zlib), data packs, and
rollback-capable patches, and testing them in an IRIX guest.

Worked examples, all built and installed this way:

| Example | What it shows | Files |
|---|---|---|
| ttr | two subsystems, terminfo compiled at install, toolchest entry, fonts, a host script driving the guest | `tools/irix-dist.sh`, `dist/irix/` in the ttr tree |
| dash | one subsystem, tagged binary, filetype rule with a system icon, toolchest, icon catalog, terminfo, man page | `Makefile` dist target, `dash.spec`, `dash.idb`, `dash.ftr`, `dash.chest` |
| chocolate-doom | separate software and data images, several launchers, own icon (`.fti`), icon catalog page | `irix/` (`dist.mk`, `.spec`, `.idb`, `.ftr`, `.fti`) |
| zlib | a library for o32, n32 and n64 with headers and archives | section 7 |

These projects live outside this repository; ask the user where their
checkouts are if you need the real files. The snippets below are enough to
build a package from scratch.

## 1. What a distribution is

| File | Contents |
|---|---|
| `<prod>` | binary product descriptor (magic `pd001V`), made by gendist from the spec |
| `<prod>.idb` | text installation database, one line per file (section 4.2) |
| `<prod>.<image>` | packed files of one image (`.sw`, `.sw64`, `.man`, `.data`, `.dev`) |

A **dist directory** holds these files. A **`.tardist`** is a plain `tar` of
them with no directory prefix, optionally with `README`s. `inst -f` takes
either.

Products contain **images**, images contain **subsystems**. A subsystem is
the unit users select, named `product.image.subsys` (e.g. `ttr.sw.term`).
Versions are **integers**: `20261020`, `301`, `10302`. Dotted versions
break upgrades.

## 2. Inspecting packages on the host (`uninst`)

`uninst` (a host-side tool, if the user has one installed) reads inst
packages on Linux:

```sh
mkdir -p /tmp/pkg && tar -xf patchSG0005734.tardist -C /tmp/pkg
cd /tmp/pkg && uninst -l patchSG0005734     # list
cd /tmp/pkg && uninst -t patchSG0005734     # verify checksums
cd /tmp/pkg && uninst -v patchSG0005734     # extract into ./usr, ./etc, ...
```

SGI's own `.idb` files are the best reference for conventions (attributes,
exitops, owners, locations). Unpack a few from your own IRIX media or patch
tardists (or ask the user whether they keep an unpacked copy) and copy their
structure.

## 3. Querying the guest

Use the guest shell from the `irix-guest-loop` skill (`tools/irix-sh.py run
'...'`). If the guest's root account has a password, pass it with
`--password` or `IRIS_ROOT_PASSWORD`; never write it into a file.
`showprods`, `versions` and `inst` page through `more`: pipe to `cat`.

```sh
versions -n ttr | cat                   # installed subsystems and versions
showprods -n ifl_eoe x_dev | cat
showfiles -- /usr/lib32/libz.so | cat   # which subsystem owns a file
versions long ttr | cat                 # every file of a product
```

Root's login shell is csh: commands passed to the guest must be
csh-compatible (no `$VAR` that may be unset, no `2>&1`).

## 4. Building a package

The build runs **in the guest**: gendist, strip and tag exist only there.
Either a `make dist` target run in the guest (dash, chocolate-doom) or a host
script that drives the guest (ttr's `tools/irix-dist.sh`) works.

### 4.1 Stage the files

Build a tree mirroring the target filesystem (`root/usr/bin/prog`,
`root/usr/share/prog/...`). For executables:

```sh
strip root/usr/bin/prog
/usr/sbin/tag 0x54545254 root/usr/bin/prog   # 32-bit tag for filetype rules
```

- Do not strip DSOs. gendist marks ELF DSOs `needrqs` so inst requickstarts
  them (`rqs`).
- Pick a tag unlikely to clash: chocolate-doom `0x444F4F4D` ("DOOM"), ttrt
  `0x54545254` ("TTRT"), dash `0x0001D450`. The idb attribute `tag(0x...)`
  makes inst reapply it.
- Stage under `/tmp` in the guest (emptied at boot; an NFS share may be
  slow or noexec) and copy inputs from the share.

### 4.2 The idb

One line per file, directory or link:

```text
<type> <mode> <owner> <group> <target> <source> <subsys> [attributes...]
f 0755 root sys usr/bin/ttrt usr/bin/ttrt ttr.sw.term
d 0755 root sys usr/share/ttr usr/share/ttr ttr.sw.wm
l 0777 root sys usr/lib32/libz.so usr/lib32/libz.so zlib.sw.lib symval(libz.so.1)
```

- Types: `f` file, `d` directory, `l` symlink (`symval(target)`).
- `target` (field 5) is the install path; `source` (field 6) is relative to
  gendist's `-sbase`. SGI omits the leading slash on both; gendist tolerates
  one in field 5 (chocolate-doom uses it).
- **Sort on fields 5 and 6** or gendist fails: `sort -k 5,6 raw.idb -o pkg.idb`.
- `d` lines are optional; inst creates parent directories.

Attributes:

| Attribute | Effect |
|---|---|
| `exitop("cmd")` | run after all files of the installation are in place |
| `postop("cmd")` | run right after this file is installed |
| `preop("cmd")` | run before this file is installed |
| `removeop("cmd")` | run when the subsystem is removed |
| `config(update)` | replace a modified config file, keep the old one as `.O` |
| `config(suggest)` | keep a modified file, install the new one as `.N` |
| `config(noupdate)` | never overwrite once installed |
| `tag(0xNNNNNNNN)` | tag the installed executable |
| `nostrip`, `norqs`, `needrqs` | strip and requickstart control |

Commands run with `/bin/sh`. `$rbase` is the install root (`/` normally,
another root with `inst -r`). Prefix every path with `$rbase`.

### 4.3 Install-time actions (tested)

**Terminfo.** Ship the source `.ti`, compile it with `tic` in an exitop. Do
not ship a precompiled entry: the compiled format belongs to the target's
tic, and IRIX's system directory is `/usr/share/lib/terminfo` (not ncurses'
`/usr/share/terminfo`). IRIX curses ignores `~/.terminfo`, so the system
directory is what every user picks up.

```text
# ttr: keep the source installed, so inst's file checks stay clean and a
# reinstall recompiles it
f 0644 root sys usr/share/ttr/ttrt.ti usr/share/ttr/ttrt.ti ttr.sw.term exitop("TERMINFO=$rbase/usr/share/lib/terminfo tic $rbase/usr/share/ttr/ttrt.ti") removeop("rm -f $rbase/usr/share/lib/terminfo/t/ttrt")

# dash: ship to /tmp and delete after compiling (inst then sees the file as
# missing in later verifications)
f 0644 root sys tmp/dash.ti tmp/dash.ti dash.sw.base exitop("tic /tmp/dash.ti && rm -f /tmp/dash.ti") removeop("rm -f /usr/share/lib/terminfo/d/dash")
```

`TERMINFO=dir tic file` writes into `dir` (IRIX tic has no `-o`). Plain
`tic file` as root writes into the system directory.

**Filetype rules.** Rebuild the desktop's database on install and removal:

```text
f 0644 root sys usr/lib/filetype/local/prog.ftr usr/lib/filetype/local/prog.ftr prod.sw.base exitop("cd $rbase/usr/lib/filetype && make") removeop("cd $rbase/usr/lib/filetype && make")
```

**Icon catalog.** Add the program to a page (Applications, Games,
DesktopTools, ...):

```text
f 0755 root sys usr/bin/prog usr/bin/prog prod.sw.base exitop("iconbookedit -add 'Category:File Name:/usr/bin/prog' -syspage Applications") removeop("iconbookedit -remove 'Category:File Name:/usr/bin/prog' -syspage Applications")
```

Copy the exact iconbookedit syntax from dash's or chocolate-doom's idb.

### 4.4 The spec

```text
product ttr
    id "To The Roots: window manager, desktop and terminal"
    image sw
        id "ttr Software"
        version 20261020
        order 9999
        subsys wm default
            id "ttr Window Manager and Desktop"
            replaces ttr.sw.wm 0 20261020
            exp ttr.sw.wm
        endsubsys
        subsys term default
            id "ttrt Terminal Emulator"
            replaces ttr.sw.term 0 20261020
            exp ttr.sw.term
        endsubsys
    endimage
endproduct
```

- `default` selects the subsystem by default.
- `replaces <name> <lo> <hi>` lets this version upgrade older ones
  (`0 maxint` or `0 <this version>`).
- `exp` names the idb subsystem whose files go into this subsystem.
- `prereq ( prod.image.subsys lo hi )` declares dependencies, e.g.
  `motif_eoe.sw.eoe`.
- Put large data in its own image (chocolate-doom `image data`) so users can
  skip it; man pages conventionally go in `image man`.
- Keep the version in one place: ttr's spec has `VERSION` placeholders that
  the build script fills with `sed`.

### 4.5 gendist and the tardist

```sh
sort -k 5,6 raw.idb -o prod.idb
/usr/sbin/gendist -sbase root -idb prod.idb -spec prod.spec -dist out [-nostrip] [-verbose]
cd out && tar cf ../prod-VERSION.tardist *
```

Other gendist flags: `-nocompress`, `-maint` (patches, section 6),
`-genidb`/`-genspec` (draft an idb or spec from a tree under `-rbase`),
`-all`.

### 4.6 Desktop files

| File | Installed to | Purpose |
|---|---|---|
| `prog.ftr` | `usr/lib/filetype/local/` | filetype rule: matches the tag, sets icon and open/drop actions |
| `prog.fti` | `usr/lib/filetype/local/iconlib/` | icon in SGI's vector format |
| `prog.chest` | `usr/lib/X11/app-chests/` | entries in the toolchest menus |
| `prog.1` | `usr/share/catman/u_man/cat1/` or `usr/share/man/man1/` | man page |

```text
TYPE ProgExecutable                          # prog.ftr
    MATCH       tag == 0x54545254;
    LEGEND      My Program
    SUPERTYPE   Executable
    CMD OPEN    $LEADER $REST
    CMD ALTOPEN launch -c $LEADER
    CMD DROP    $TARGET $SELECTED
    ICON        { include("../system/iconlib/ttyExec.fti"); }

Menu Desktop                                 # prog.chest
{
    "My Program"    f.checkexec "/usr/bin/prog &"
}
```

## 5. Installing and removing for tests

```sh
inst -a -f /tmp/prod-dist/out -I prod      # install, no questions
inst -a -f /tmp/prod-dist/out -I prod.sw.term   # one subsystem
inst -a -R prod                            # remove
inst -f /path/prod-VERSION.tardist         # interactive
```

`-a` runs without menus; `-I` and `-R` take products or subsystems. The
output lists every exitop and removeop that ran. Check with
`versions -n prod | cat`, `showfiles`, and by running the program from its
installed path in a fresh login. Remove again and check that removeops
cleaned up (terminfo entries, filetype database).

## 6. Patches (rollback-capable maintenance products)

A subsystem with the `patch` attribute and a `follows` rule, built with
`gendist -maint`, is an SGI patch: inst saves the files it replaces in
`/var/inst/patchbase/` and restores them when the patch is removed.

- The patch subsystem **must** have `follows <base_prod>.<image>.<subsys>
  <min_ver> <max_ver>` matching the installed base (check `versions -n`).
- If another patch already sits on that base (e.g.
  `patchSG0005734.ifl_eoe_sw.c++` at `1319999907`), either
  `replaces patchSG0005734.ifl_eoe_sw.c++ 0 maxint`, or follow the base
  (`follows ifl_eoe.sw.c++ 1317906130 maxint`) so inst archives the
  currently active files.
- Copy the spec structure from a real SGI patch unpacked with `uninst`.
- Files owned by an SGI subsystem (`showfiles` tells which) can only be
  replaced cleanly by a patch. A standalone product installing the same
  paths makes inst report conflicts; otherwise install under new names.

## 7. Libraries for several ABIs (example: zlib)

o32 goes to `usr/lib`, n32 to `usr/lib32`, n64 to `usr/lib64`, headers to
`usr/include`. n64 gets its own image so 32-bit systems can skip it:

```text
product zlib_modern
    id "Modernized zlib 1.3.2 Compression Library for IRIX 6.5"
    image sw
        id "zlib 1.3.2 Runtime Shared Libraries (o32/n32)"
        version 10302
        subsys lib default
            id "zlib 1.3.2 Shared Libraries"
            replaces zlib_modern.sw.lib 0 maxint
            exp zlib_modern.sw.lib
        endsubsys
    endimage
    image sw64
        id "zlib 1.3.2 64-bit Runtime Shared Library (n64)"
        version 10302
        subsys lib default
            id "zlib 1.3.2 64-bit Shared Library"
            replaces zlib_modern.sw64.lib 0 maxint
            exp zlib_modern.sw64.lib
        endsubsys
    endimage
    image dev
        id "zlib 1.3.2 Development Headers and Static Archives"
        version 10302
        subsys lib default
            id "zlib 1.3.2 Headers and Static Libraries (o32/n32/n64)"
            replaces zlib_modern.dev.lib 0 maxint
            exp zlib_modern.dev.lib
        endsubsys
    endimage
endproduct
```

## 8. Gotchas

- gendist fails unless the idb is sorted on fields 5 and 6.
- Versions are integers.
- Every exitop and removeop path needs `$rbase`.
- `showprods`, `versions` and `inst` page their output: pipe to `cat`.
- In agent sessions a host shell's `cd` may fail; use absolute paths.
- Backslashes inside backticks in host scripts are halved. Get file lists
  from make (`make -s print-target`) instead of parsing Makefiles with sed.
- Test the installed program from its installed path: programs that find
  data relative to themselves must work from `/usr/bin`.
