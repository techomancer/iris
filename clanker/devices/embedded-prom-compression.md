# Compressing the embedded PROMs

Which compression the PROMs built into the binary use. Each machine inflates only its own PROM
when it starts, so every method below stores one stream per PROM unless it says otherwise.

## What was compared

The input was what `prom_image::decode` returns for each of eight PROMs (IP22, IP24, IP28, IP30,
IP32, IO6, IP27, IP35 6.210): 6,042,214 bytes. The IO6 is unwrapped here, but iris stores it as
published (365,328 bytes): it is the BaseIO flash's contents, and the IP27 PROM unpacks the
container from the flash itself (given the unwrapped payload it rejects the magic and uses its own
copy). Its payload is already gzip-compressed, so it deflates to 360,783 bytes, and the files in
`data/proms` total 3,332,630 bytes, 502 fewer than the deflate row. Each method was measured with a Rust decoder in a release build (fat LTO,
panic=abort), best of 7, on an AMD EPYC 7663 under load, and every stream was decoded back and
checked.

IP27 and IP35 each carry a gzip member (345 KB and 656 KB compressed). It looks random to any outer
compressor and cannot be re-encoded without changing what the guest sees, so it caps every method.

| Method | Stored, 8 PROMs | Decode IP35 | Added to binary | New dependency |
|---|---|---|---|---|
| deflate-9, `flate2` | 3,333,132 | 4.1 ms | none | no |
| deflate via zopfli at pack time | 3,208,903 | 4.7 ms | none | packer only |
| LZMA2, `lzma-rust2` | 2,804,092 | 16.9 ms | +25 KB | yes |
| zstd-19, `ruzstd` | 3,045,975 | 5.9 ms | +68 KB | yes |
| zstd-19, `zstd` (C) | 3,045,975 | 1.7 ms | +157 to +419 KB | yes |
| sibling dictionary, zstd | 2,699,485 | 5.7 ms (worst 13.3 ms) | as ruzstd | yes |
| sibling dictionary, LZMA2 | 2,488,808 | 21.8 ms (worst 24.9 ms) | as lzma-rust2 | yes |
| all PROMs as one LZMA2 stream | 2,433,549 | 76 ms, every machine | as lzma-rust2 | yes |
| all PROMs as one zstd stream | 2,639,168 | 28 ms, every machine | as ruzstd | yes |

A sibling dictionary compresses one PROM against another's bytes: a machine may decode two. It
helps the Indy and Indigo2 family (IP22 against IP24 saves 135 KB), IP27 with IP35 and IO6 with
IP30. A trained dictionary was larger than none.

## The choice

Deflate per PROM through `flate2`: nothing new, 4 ms for the largest PROM, the simplest to build.
LZMA2 would save about 530 KB (16%) for about 13 ms more on IP35 and one more dependency; that is
the one trade that moves a number. Zopfli would save 124 KB with no runtime cost, for a dependency
in the packer. Dictionaries add an ordering rule and a two-step start, and the one-stream methods
make every machine decode all eight.

## Small boards

Not measured: no Raspberry Pi or other ARM board was available. A Pi 5 is probably 1.5 to 2.5
times slower per decode, which would put deflate at 6 to 10 ms for IP35 and LZMA2 at 25 to 42 ms.
That is an estimate from a published zstd result and a Geekbench 6 score, not a measurement.

## Keeping the PROMs in git

Measured on two revision pairs, IP35 6.210 to 6.211 and the two IP28 6.2 reva builds, neither
shipped here (`git gc --aggressive` each time). A revision rewrites most of a PROM, so a compressed
file adds about its own size to the repository (IP35 0.9 MB, IP28 0.2 MB) and no delta helps. A
single blob is worse, since a change moves everything after it, and Rust arrays are 1.6x to 1.9x
bigger with diffs nobody can read. One file per PROM, replaced only when its PROM changes,
embedded with `include_bytes!`, keeps a change to the one file.
