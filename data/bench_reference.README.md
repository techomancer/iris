# data/bench_reference.json

What a benchmark result gets compared against in the GUI. **Starts empty, and
empty is a normal state** — a machine with no row here simply gets "reference
statistics not gathered for this platform" instead of a comparison. Nothing is
uploaded, nothing is downloaded, and there is no user-writable override: it is a
static file updated by hand when someone measures a machine worth recording.

## Adding a row

```sh
make -C bench                      # build the guest binary
cargo build --release --bin iris-bench
./target/release/iris-bench run --label my-machine

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

Both are recorded because the two engines differ by roughly 4x, so a row without
them cannot be compared with anything. Note the Mac App Store build forces
`IRIS_NO_JIT=1` (`iris-gui/src/main.rs`) — the sandbox only permits `MAP_JIT`
pages and Cranelift does not use them — so rows meant for comparison against a
store build must be `"engine": "interp"`.
