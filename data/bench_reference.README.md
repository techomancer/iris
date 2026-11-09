# data/bench_reference.json

What a benchmark result gets compared against in the GUI. **Starts empty, and
empty is a normal state** — a machine with no row here simply gets "reference
statistics not gathered for this platform" instead of a comparison. Nothing is
uploaded, nothing is downloaded, and there is no user-writable override: it is a
static file updated by hand when someone measures a machine worth recording.

## Adding a row

```sh
cargo build --release --bin iris-bench
./target/release/iris-bench run --label my-machine     # NOT --quick

./target/release/iris-bench reference \
    --id m1-max-interp \
    --label "MacBook Pro (M1 Max) — interpreter" \
    --into data/bench_reference.json
```

`reference` with no `--from` uses the newest result in `bench/build/results/`.
Without `--into` it prints the row to stdout for pasting.

## `suite_id`

The blake3 of the guest binary the numbers were measured against. Reference
figures only mean something against the exact suite that produced them: add or
change a kernel and every stored number silently becomes a comparison between
two different workloads.

So the merge refuses when a result's suite disagrees with a populated table, and
the GUI treats a mismatch exactly like an empty table. **If the suite changes,
every row has to be re-measured** — which is the honest cost of having reference
numbers at all, and the reason the file is small on purpose.

An empty table adopts the suite of the first row merged into it.

## `cpu` and `engine`

Both are recorded because the engines differ by roughly 4x, so a row without
them cannot be compared with anything. The engine comes from the emulator's own
`build features:` banner: `interp`, `lightning`, `jitv2`, or `jitv2-lightning`.
Lightning is its own engine: it compiles out breakpoints and traceback to run
faster, so a lightning result does not match an `interp` row.

The Mac App Store build forces `IRIS_NO_JIT=1` (`iris-gui/src/main.rs`) — the
sandbox only permits `MAP_JIT` pages and Cranelift does not use them. The store
build is lightning (`.github/workflows/appstore.yml` builds it with `--features
iris/lightning,appstore`), so rows meant for comparison against a store build
should carry `"engine": "lightning"`. The engine is read from features alone,
so a jitv2 build run with `IRIS_NO_JIT=1` is still labelled `jitv2`.

## Full runs only

`reference` refuses a result recorded with `--quick`. A shortened run is still
accurate — every kernel ran and verified — but its rates come from a single
timed pass at 30% of the usual target time, and this table is what every other
machine gets compared against. Re-run without `--quick`.
