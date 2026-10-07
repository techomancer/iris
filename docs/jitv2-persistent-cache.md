# jitv2 persistent code cache — design

Implemented in `src/cpu/jitv2/pcache.rs`; landed on 2026-09-29 (`36ec12b`).
The September 24 measurements below predate that landing. Where the implementation departs from the plan below, the
section says so.

Update, 2026-10-01: the toggle is `[jitv2] cache`/`cache_dir` in
`iris.toml` (iris-gui **General → Persistent JIT code cache** since
2026-10-02), which
`Jitv2Config::apply_env` (`src/config.rs`) turns into the `IRIS_JIT_CACHE`/
`IRIS_JIT_CACHE_DIR` env vars described below. False/blank config values
preserve existing environment values; enabled/nonempty config replaces them.
The supported configuration inventory is in [FEATURES.md](../FEATURES.md).

The R10000 JIT recompiles the same pages in every run. This keeps compiled
pages on disk, keyed by what they were compiled from, so a later run loads
them instead of running Cranelift again. It is also the storage an
ahead-of-time compiler would fill (the last section).

## Why: measured, not assumed

Two identical sessions on an IP28 (R10000) machine (boot to login, two
MIPSpro compiles, two
`ls -lR /usr/include`, small utilities), every compile logged with its page
content hash, FR mode and entry set (`IRIS_JIT_HASHSTATS_LOG`):

| | run 2's compiles | of compiled instructions |
|---|---|---|
| page content already compiled in run 1 | **97.1%** | 97.0% |
| a run-1 compile covers the entries run 2 wanted | 82.7% | 79.8% |
| run 1's entries merged per page (a cache that grows its variants) | **85.7%** | 83.0% |
| first quarter of run 2 (boot), merged | **92.9%** | 91.4% |

Cost of what a hit replaces: 5,500 compiles took **122 s of Cranelift time,
22 ms per page** (4 compile threads). A warm cache saves on the order of
100 CPU-seconds per session, concentrated where it hurts: boot, and the first
run of a program, which today waits behind the compile queue.

Within one session, by contrast, no page's content is ever compiled at a
second physical address (0 of ~6,500): IRIX keeps a program's text in the
same frames. The reuse is across runs and across mega-flushes, which is what a
persistent cache captures.

Footprint: one run compiled 3,053 distinct (page, FR) pairs.

## What a compiled page depends on: the key

A compiled page is one Cranelift function with a dispatch switch over its
entry points. It is fully determined by:

1. **The page's bytes.** v1 keys on all 1024 words and stores them, so a hit
   is verified with a 4 KB compare; no hash collision can serve wrong code.
2. **Its entry set.** A blob serves any request whose entries are a subset of
   the blob's (the switch has a case for each).
3. **FR mode**, **ISA level** (MIPS IV), **CPU model**.
4. **Codegen configuration:** direct memory on or off, inline memory on or
   off, cache geometry, instruction budgets, opt level, the per-category
   enables (`j2 alu|fpu|...`), CP0/atomics in-region, lockstep/developer
   features. All of it goes into a fingerprint computed **per compile**, since
   several are runtime toggles.
5. **Build identity:** a BLAKE3 hash of the running executable. Any rebuild is
   a new cache namespace; nothing is ever served across builds.
6. **Host ISA:** Cranelift's target flags (detected CPU features).

The physical page is **not** an input: branch targets are page-relative,
`j`/`jal` resolve at run time, and the `PhysicalCodePage` pointer codegen
receives is bookkeeping, never emitted.

Correctness does not need Cranelift to be deterministic: a blob only has to be
*a* correct compilation of identical inputs under an identical configuration.
The real risk is a hidden input missing from the key, which is why the
fingerprint is strict and the key is conservative (whole page, whole build).

## Position independence

Compiled code can bake two kinds of host address:

- **Rust hook pointers** (`JitConsts::hook_addr`): read out of the core at
  compile time and emitted as call targets. They move with every launch
  (ASLR). **Only under `IRIS_BAKE_HOOKS=1`**: baking measured as a code-size
  loss, so it is off by default, and codegen loads the pointer from the core.
- **Shared memory-helper addresses** (`emit_mem_helper_call`): they live in
  the JIT arena. Off by default (`IRIS_MEM_HELPERS` enables them).

So default compiled code is already position independent: the core pointer is
a function argument, never baked. `Codegen::cache_fingerprint` refuses (no
caching) when either switch would put an address in the code.

Correction: an earlier draft reported a cost for `IRIS_JIT_PIC=1` (median
+1.1%, geometric mean +2.7% on jitcov). With `IRIS_BAKE_HOOKS` unset, the
published core address is read by nothing but `hook_addr`, which returns
before using it, so the two builds emit the same code. Those numbers were
run-to-run variance, not a cost.

Guard: a blob is stored only if Cranelift reports **no relocations** for it
(for example a library call for an FP operation). Such a page simply isn't
cached.

## Storage

As implemented: no index file. The layout is
`<base>/<build-id>/<fingerprint>/<page-hash>-<fr>/<entries-hash>.jc`, and a
lookup lists one page directory (a failed `open` on a never-seen page). Blobs
carry a BLAKE3 checksum; a new variant deletes the ones it covers. The plan
as first written:

- `$IRIS_JIT_CACHE_DIR`, default the user cache directory's `iris/jitv2/<build-id>/` (`~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows, `$XDG_CACHE_HOME` or `~/.cache` elsewhere)
  (off the project drive; per build).
- One file per blob, named `<page-hash>-<fr>-<entries-hash>.jc`: a header
  (magic, format version, fingerprint, FR, entry bitmap, used-word bitmap,
  code alignment and length), then the 4 KB page, then the code.
- Written by a background thread via temp file + rename, so a crash never
  leaves a torn blob; a blob that fails its header or length check is ignored
  and deleted.
- At startup the directory is scanned into an index
  `(page-hash, fr) -> [blob]`. About 3,000 files for a session: milliseconds.
- Pruning: keep the newest three build directories, and cap total size
  (LRU by access time).

## Lookup and insert

In `comp.rs`'s deferred path, after `prepare_multi_entry_compile` (snapshot and
walk, both cheap):

1. Hash the page's words. Find a blob with the same FR mode whose entries
   cover this compile's covered entries.
2. **Hit:** compare the stored 4 KB with the snapshot; on any difference it is
   a miss. Otherwise `declare_function` + `define_function_bytes` (Cranelift
   copies the bytes through the same `PagedArenaMemoryProvider`), then the
   unchanged placeholder / seal / publish path. It publishes the blob's
   entries (a superset, all valid for these bytes) and stages the churn
   snapshot from the blob's used-word mask.
3. **Miss:** compile the *union* of the requested entries and the best cached
   variant's, so variants converge towards "every entry this page ever
   needed" (the 97% ceiling above). Hand the bytes to the writer thread.

Nothing else changes: generation checks, the seal queue, churn avoidance and
denials behave as for a compile.

## Verification plan

1. The page compare, fault-injected: `IRIS_BREAK=jitcache-skip-compare`
   serves a blob for a page with one word changed. It must fail a test that
   passes with the compare, so the compare is known to be load-bearing.
   (As run: a full-page hash never collides by itself, so this also needs
   `jitcache-weak-key`; see Results.)
2. jitcov cold vs warm: all 188 kinds agree in both. Warm timings equal cold
   ones, since the code is the same.
3. IP28 boot to login, cold then warm: hit rate (expect about 90% at boot),
   compile time, and time to login.
4. The first-program backlog: `jitcov nowarm` right after boot, cold vs warm.
5. A rebuild never serves an old blob (a new build id, a new directory).

## Results (2026-09-24, IP28 / R10000, 4 compile threads)

Workload per session: boot to login, a MIPSpro compile of jitcov, `jitcov
100000 nowarm` (the first program after boot), `ls -lR /usr/include` twice, a
loop of small utilities, `jitcov 100000`. One cold session (empty cache), then
two warm ones.

| | cold | warm | warm 2 |
|---|---|---|---|
| lookups served from the cache | 15% (in-run) | 86% | 95% |
| Cranelift time, all threads | 143 s | 63 s | 27 s |
| boot to `login:` | 42 s | 39 s | 39 s |
| first program: jitcov routines still > 10 ns | 27 of 188 | 3 | 3 |
| first program: sum of the 188 timings | 1060 ns | 205 ns | 195 ns |

A cached page loads in about 8 µs; compiling it takes about 22 ms. No compile
was refused for relocations. In every jitcov run, cold and warm, the 186
deterministic kinds give the same checksums as the last no-cache run. `sc` and
`scd` count successful store-conditionals, which an interrupt between `ll` and
`sc` makes fail, so they differ by a few counts in every run, with or without
the cache. (An earlier version of this section claimed all 188 agree; that was
a misreading.)

The page compare, fault-injected with a guest program built for it
(`cpu-tests/jitcov/irix-kc`, `kc K`: 200 `addiu a0,a0,K`, so runs with different K differ only in
immediates):

- `IRIS_BREAK=jitcache-weak-key` keys pages on each word's top 16 bits only.
  The runs collide in the cache; the compare rejected 8 of them and every
  result was correct.
- Adding `jitcache-skip-compare` serves whatever collides. The guest could not
  finish the MIPSpro compile that precedes the test.

So the compare is load-bearing, and it holds. (The SMC test could not serve
here: its function is three words and never exercised the cache.)

A rebuild gets a new build id and so an empty cache namespace; a new build's
first session showed the cold hit rate.

Size: 210–270 MB per build (about 3,300–4,400 blobs of 64 KB on average, most
of it code), so up to about 800 MB with three builds kept. No size cap yet.

The warm sessions first measured here showed a loop that the cache exposed but
did not cause: one kernel page prepared thousands of times at one generation
(0x20010: 6,734 FR0 compiles straight after 3 FR1 ones), each `publish`
refused. Kernel pages run under whichever FR mode the interrupted process
uses. An FR1 compile was published, the next request came from an FR0
context and re-pinned the page, and the FR0 compile was refused as "already
covered at this generation": that check ignored the FR mode. The page was
left pinned FR0 with FR1 code and an FR1 churn snapshot, so every later
request failed the churn skip on the mode, recompiled and was refused again,
until a mega-flush. Arrivals at denylisted offsets (kernel entries the JIT
excludes, hit on every exception) supplied the requests. Without the cache
each round is a 22 ms compile, which throttles it; with the cache it is an
8 µs load, so it ran about 10,000 extra rounds per session.

Fixed in two parts: `publish` records the FR mode of the installed code and
treats a compile for the other mode as never redundant (replacing the
advertised entries, keeping denials); and the dispatch gate no longer
requests a compile for a denylisted offset while the page's bytes are
unchanged. Five sessions afterwards (one cold, three warm, one without the
cache): at most 48 compiles of any page per session, 8,600-9,100 in total
(16,500-17,000 while looping), one mega-flush each, and the same warm
results as above (hits 80-96%, Cranelift time 45, 18 and 14 s). R4400 and
R5000 Indy boots still reach login.

## Risks

- **A hidden input left out of the fingerprint** serves code compiled for a
  different configuration. Mitigations: the whole-build id, a conservative
  fingerprint, and `IRIS_JIT_CACHE=off` as a kill switch.
- **Disk-resident executable code:** the cache directory is the user's own;
  blobs are never shared between machines or users.
- **Size:** roughly 3,000 blobs per build for this workload.

## Implementation steps

1. `IRIS_JIT_PIC` becomes the default when the cache is on (done as a switch).
2. Capture bytes, alignment and relocations after `define_function`; store
   only relocation-free blobs.
3. The blob format, writer thread and startup index.
4. Hit path via `define_function_bytes`, with the page compare.
5. Union-on-miss.
6. Verification 1–5.

## Towards AOT

The key is page content, not where or when a page was compiled, so an offline
tool can fill the same cache: walk the text pages of the IRIX kernel (`/unix`)
and the shared libraries (quickstart-prelinked at fixed addresses, so their
pages are byte-identical in memory), compile each with a superset of plausible
entry points (symbols, branch targets, return sites), and write blobs. The
first boot of a fresh build would then start warm.
