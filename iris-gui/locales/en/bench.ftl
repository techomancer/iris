# Benchmark tab.

## Controls

bench-heading = Benchmark
bench-subtitle =
bench-run-mac = Benchmark this Mac
bench-run-pc = Benchmark this PC
# $secs is a rough run time in seconds.
bench-run-tip = Takes about { $secs } seconds.
bench-cpu-tip = Scores from different CPUs aren't comparable.
bench-quick = Quick run
bench-quick-tip = Runs each kernel once with shorter timing. Accuracy is unaffected; speed figures are a little noisier.
bench-stop = Stop
bench-machine-running = Stop the machine to run a benchmark.

## Progress

bench-starting = Starting…
# $kernel is the name of the test running now, such as int/alu.
bench-progress = { $kernel } ({ $done } of { $total })
# Times in seconds.
bench-time-total = { $elapsed } s elapsed, about { $total } s in total
bench-time-left = { $elapsed } s elapsed, about { $left } s left
bench-stopped = Stopped.
# $error is the reason, in English.
bench-failed = The benchmark didn't finish: { $error }

## Results

bench-results = Results
bench-dmips = Emulated Indy
bench-dmips-value = { $value } DMIPS
bench-dmips-tip = Dhrystone 2.1 score. A real 150 MHz Indy scores about 130.
bench-mips = Emulator speed
bench-mips-value = { $value } MIPS
bench-mips-tip = Guest instructions run per second.
bench-accuracy = Accuracy
bench-accuracy-value = { $percent }% ({ $matched } of { $checked })
bench-accuracy-tip = Tests whose result matched a natively computed checksum. Anything below 100% is an emulator bug.
# $ratio is a number such as 1.25; $reference names the reference machine.
bench-ratio = { $ratio }× vs { $reference }
bench-no-reference = No reference results for this platform yet.
# $reason is in English, such as "30% timing, 1 repeat".
bench-caveat-quick = Quick run ({ $reason }). Accuracy is unaffected.
bench-caveat-interpreter = This build has no MIPS JIT. JIT builds score about four times higher, so these numbers aren't comparable with theirs.
# $kernels is a comma-separated list of test names.
bench-caveat-exceptions = Unexpected exceptions in { $kernels }. Those figures aren't reliable.
bench-save-report = Save report…
bench-save-report-picker = Save benchmark report
bench-save-report-filter = JSON

## Measured machine

bench-machine = Measured machine
bench-machine-cpu = CPU
# $prid is the processor ID register value, in hex.
bench-machine-cpu-value = { $cpu } ({ $prid })
bench-machine-cpu-rev-value = { $cpu } rev { $rev } ({ $prid })
bench-machine-l1 = L1 cache
# Instruction and data cache sizes, with their line sizes in bytes.
bench-machine-l1-value = { $icache } I ({ $iline } B lines) / { $dcache } D ({ $dline } B lines)
bench-machine-l2 = L2 cache
bench-machine-l2-absent = None
bench-machine-l2-value = { $size }, { $line } B lines
bench-machine-l2-unsized = Present, { $line } B lines, size unknown
bench-machine-memory = Memory
# $banks is a list of bench-machine-bank entries.
bench-machine-memory-value = { $total } MB   { $banks }
# $base is the bank's start address, in hex.
bench-machine-bank = bank { $index }: { $size } MB at { $base }
bench-machine-board = Board
bench-machine-board-value = SYSID { $sysid }
bench-machine-emulator = Emulator
bench-machine-no-features = No optional features
bench-machine-host = Host
bench-machine-host-value = { $cpu } · { $os } { $arch } · { $cores ->
    [one] 1 core
   *[other] { $cores } cores
}
bench-machine-unknown-cpu = Unknown CPU

## Console output

bench-details = Console output
bench-details-empty = No output yet.

## Developer tools

bench-dev = Developer tools
bench-dev-subtitle =
bench-dev-matrix = Full matrix
bench-dev-matrix-tip = Builds and runs R4400 and R5000, with and without the JIT. Takes tens of minutes and needs cargo.
bench-dev-host = Measure this host
bench-dev-host-tip = Runs the same tests compiled natively, for comparison.
bench-dev-not-built = iris-bench isn't built. Run cargo build --release --bin iris-bench.
bench-dev-open-results = Open results folder
# $path is the iris-bench program, $error the system's error text.
bench-dev-start-failed = Couldn't start { $path }: { $error }
bench-dev-matrix-finished = Matrix finished.
bench-dev-matrix-failed = Matrix failed.
bench-dev-host-finished = Host measurement finished.
bench-dev-host-failed = Host measurement failed.
bench-running = A benchmark is already running.
bench-save-failed = Couldn't save the report: { $error }
