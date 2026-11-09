#!/bin/sh
# Fail if the interpreter's hot paths regressed in a release binary. Needs a
# lightning build with fat LTO (the release profile) and GNU nm and objdump on
# an ELF binary: without fat LTO, calls into std go through the GOT and the
# check reports the binary unreadable. See
# clanker/perf/hot-path-inlining-must-be-explicit.md.
# Usage: tools/hot-path-check.sh target/release/iris
set -eu
bin=${1:?usage: hot-path-check.sh BINARY}
test -f "$bin"

# A fetch that must inline has its own symbol. The functions that follow are
# #[inline(never)] in the source, so they exist whatever the inliner decides;
# if one is missing, a type was renamed and the check above passes vacuously.
if nm -C "$bin" | grep -E '(CpuCache|ShadowCache|DecodeCache).*>::fetch$'; then
    echo "FAIL: a hit-path fetch is out of line" >&2; exit 1
fi
for f in fill_l1i_line index_store_data fetch_slow; do
    nm -C "$bin" | grep -q "::$f" || { echo "FAIL: anchor $f not found" >&2; exit 1; }
done

# No formatting call in MipsExecutor code except the cold methods that print by
# design. Those also act as the control: if none shows a formatting call, the
# disassembly cannot show them and the result means nothing.
objdump -d -C --no-show-raw-insn "$bin" | awk '
/^[0-9a-f]+ <.*>:$/ {
    cur = $0
    cold = cur ~ /::(report_[a-z_]*error|fault_note|post_refused_store|idle_profile_report|backtrace_from|jitv2_monitor_pfn|handle_exception|debug_|monitor_)/
    hot = cur ~ /MipsExecutor/ && !cold
    ctl = cur ~ /MipsExecutor.*::(report_[a-z_]*error|handle_exception)/
    next
}
/\t(call|jmp)[lq]? .*<(std::io::stdio::_e?print|.*::write_fmt|core::fmt::|alloc::fmt::format|<[ui](8|16|32|64|size)>::_fmt)/ {
    if (ctl) seen++
    if (hot) { bad++; print cur; print "   " $0 }
}
END {
    if (!seen) { print "FAIL: unreadable binary (no formatting call in the control)"; exit 1 }
    if (bad) { print "FAIL: " bad " formatting calls in hot code"; exit 1 }
}'
