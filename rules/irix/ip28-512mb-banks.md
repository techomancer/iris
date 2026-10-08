# IP28 512 MB banks

For 1 GB of RAM on IP28, configure two 512 MB banks:

```toml
banks = [512, 512, 0, 0]

[machine]
profile = "indigo2_ip28"
cpu = "r10000"
```

IP28 has an embedded PROM fallback; use `prom` to override it. In the GUI,
select the IP28 profile and the **1024 MB** preset, or set banks 0 and 1 to **512 MB**.
Stop and start the machine to apply the configuration.

IP28 uses a 16 MB memory configuration (MEMCFG) granule. Its five-bit size
field can represent 32 units: field 31 describes 512 MB. The PROM sizes
this bank as two 256 MB subbanks. The installed-size table must include
`(31, 1)` for 512 MB. Without that entry, `parse_memcfg` treats the
installed bank as absent even when the PROM sets its valid bit. The existing decoder then produces a `0x1fffffff` address
mask and a `0x20000000` slot size when the subbank bit is set. During
POST, `0x3f20` must alias at 256 MB; `0x7f20` must expose the full 512 MB.
With the subbank bit set, the mirror period is the total installed size.
Deriving it by doubling the size field would incorrectly produce 1 GB
for this bank and disable its direct host mapping.

The [Linux IP28 bank decoder](https://android.googlesource.com/kernel/common/+/refs/heads/android-trusty-3.10/arch/mips/sgi-ip22/ip22-mc.c)
uses the same size calculation: `(size_field + 1) << 24`.

IP22 and IP24 continue to reject banks above 128 MB. IRIX's policy for the
fourth bank is unchanged; this configuration populates only two banks.

Regression coverage checks configuration validation, PROM-style aliasing,
MEMCFG encoding, physical bank slot routing, bus/direct memory agreement
across the 256 MB boundary and bank ends, and GUI preset distribution.

## Live validation

On 2026-10-04, `indigo2_prom_ip28.bin` completes POST with
`MEMCFG0 = 0x7f207f40` for `[512, 512, 0, 0]`. Both banks have a
`0x1fffffff` address mask and a `0x20000000` slot size. IRIX 6.5 boots
from a copied disk with overlays disabled. Over SSH, `hinv -t memory`
reports `Main memory size: 1024 Mbytes`; `uname -a` identifies
`IRIX64 blueindigo 6.5 10070055 IP28`.

The targeted regressions pass with both the default build and `jitv2`.
The GUI preset test and compilation check also pass.
