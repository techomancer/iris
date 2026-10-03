#!/usr/bin/env python3
"""
decode_vc2_ram.py - Decode VC2 RAM dump from 'vc2 ramdump' monitor command.

Usage:
    iris$ vc2 ramdump > vc2.dump
    python3 tools/decode_vc2_ram.py vc2.dump

Output:
  - VT (Video Timing) table: each line entry decoded with signal names,
    durations in pixel clocks, and cumulative pixel offsets.
  - DID table: run-length entries per scanline.

VC2 VT encoding (from VC2 datasheet):
  Each state run is 1 or 2 words:
    Word 1: bit15=EOL, bit14:8=duration(SRUN, 7 bits, units of 2px), bit7=SB/SC_absent, bit6:0=state_A
    Word 2 (if present): bit15=EOL, bit14:8=state_B (7 bits), bit7=1 (always), bit6:0=state_C

  State A (bit positions, active-low _N signals):
    bit0 = VIS_LN_VC_N
    bit1 = HPOS_VC_N
    bit2 = DSPLY_EN_RO_N
    bit3 = SER_EN_RO_N
    bit4 = TX_REQ_REX_N
    bit5 = CSYNC_DAC_N
    bit6 = CBLANK_DAC_N

  State B (bit positions):
    bit0 = HBLANK_AB_N
    bit1 = EOF_AB_N
    bit2 = CBLANK_CMAP_N
    bit3 = SET_TSC_REX_N
    bit4 = ODDFIELD_VC_N
    bit5 = EOF_VC_N
    bit6 = VPOS_VC_N

  State C (bit positions):
    bit0 = VERT_INT_REX_N
    bit1 = VSYNC_ARC_N
    bit2 = HSYNC_ARC_N
    bit3 = CSYNC_ARC_N
    bit4 = VERT_STAT_GIO_N
    bit5 = spare
    bit6 = CBLANK_XMAP_N
"""

import sys
import argparse

STATE_A_NAMES = [
    "VIS_LN",    # bit0
    "HPOS",      # bit1
    "DSPLY_EN",  # bit2
    "SER_EN",    # bit3
    "TX_REQ",    # bit4
    "CSYNC_DAC", # bit5
    "CBLANK_DAC",# bit6
]

STATE_B_NAMES = [
    "HBLANK_AB",   # bit0
    "EOF_AB",      # bit1
    "CBLANK_CMAP", # bit2
    "SET_TSC",     # bit3
    "ODDFIELD",    # bit4
    "EOF_VC",      # bit5
    "VPOS",        # bit6
]

STATE_C_NAMES = [
    "VERT_INT",      # bit0
    "VSYNC_ARC",     # bit1
    "HSYNC_ARC",     # bit2
    "CSYNC_ARC",     # bit3
    "VERT_STAT_GIO", # bit4
    "spare",         # bit5
    "CBLANK_XMAP",   # bit6
]

def active_signals(val, names):
    """Return list of asserted (active-low = bit clear) signal names."""
    result = []
    for i, name in enumerate(names):
        if not (val & (1 << i)):
            result.append(name)
    return result

def inactive_signals(val, names):
    """Return list of deasserted signal names."""
    result = []
    for i, name in enumerate(names):
        if val & (1 << i):
            result.append(name + "_n")
    return result

def decode_state_run(w1, w2=None):
    """Decode one state run. Returns (duration_px, state_a, state_b, state_c, eol)."""
    eol = bool(w1 & 0x8000)
    duration = (w1 >> 8) & 0x7F   # in units of 2 pixel clocks
    sb_sc_absent = bool(w1 & 0x0080)
    state_a = w1 & 0x7F

    state_b = 0
    state_c = 0
    if not sb_sc_absent and w2 is not None:
        state_b = (w2 >> 8) & 0x7F
        state_c = w2 & 0x7F

    duration_px = duration * 2
    return duration_px, state_a, state_b, state_c, eol

def sig_str(state, names, show_inactive=False):
    active = active_signals(state, names)
    parts = [f"\033[1m{s}\033[0m" for s in active]
    if show_inactive:
        inactive = [names[i] for i in range(7) if state & (1 << i)]
        parts += [f"\033[2m{s}!\033[0m" for s in inactive]
    return " ".join(parts) if parts else "-"

def decode_line(ram, ptr, verbose=False):
    """
    Decode one VT line starting at ptr.
    Returns list of (px_start, duration_px, state_a, state_b, state_c, eol) and next_ptr.
    """
    entries = []
    px = 0
    state_b, state_c = 0, 0
    safety = 0
    while True:
        if ptr >= len(ram):
            break
        w1 = ram[ptr]; ptr += 1
        eol = bool(w1 & 0x8000)
        duration = (w1 >> 8) & 0x7F
        sb_sc_absent = bool(w1 & 0x0080)
        state_a = w1 & 0x7F

        if not sb_sc_absent:
            if ptr >= len(ram):
                break
            w2 = ram[ptr]; ptr += 1
            if w2 & 0x8000:
                eol = True
            state_b = (w2 >> 8) & 0x7F
            state_c = w2 & 0x7F
        # else reuse previous state_b, state_c

        duration_px = duration * 2
        entries.append((px, duration_px, state_a, state_b, state_c))
        px += duration_px
        if eol:
            break
        safety += 1
        if safety > 4096:
            print("  WARNING: line safety limit hit")
            break

    # next line pointer follows the EOL entry
    if ptr < len(ram):
        next_ptr = ram[ptr]
        ptr += 1
    else:
        next_ptr = 0xFFFF

    return entries, next_ptr, ptr

def print_line(entries, line_idx=None, indent="  "):
    prefix = f"  Line {line_idx}: " if line_idx is not None else indent
    print(f"{indent}{'px_start':>8} {'dur':>5} | A: {'SIGNALS (active=asserted)':40} | B: {'':30} | C: {'':30}")
    print(f"{indent}{'--------':>8} {'---':>5} | {'':41} | {'':31} | {'':31}")
    for (px, dur, sa, sb, sc) in entries:
        a_str = " ".join(active_signals(sa, STATE_A_NAMES)) or "-"
        b_str = " ".join(active_signals(sb, STATE_B_NAMES)) or "-"
        c_str = " ".join(active_signals(sc, STATE_C_NAMES)) or "-"
        print(f"{indent}{px:8d} {dur:5d} | A: {a_str:40s} | B: {b_str:30s} | C: {c_str:30s}")

def decode_vt(ram, frame_ptr, verbose=False):
    print(f"\n=== VIDEO TIMING TABLE (frame_ptr=0x{frame_ptr:04x}) ===")
    curr = frame_ptr
    seq_idx = 0
    total_lines = 0
    safety = 0

    while curr + 1 < len(ram):
        line_seq_ptr = ram[curr]
        line_count   = ram[curr + 1]
        if line_count == 0:
            print(f"\n  [seq {seq_idx}] End of frame (line_count=0)")
            break
        print(f"\n  [seq {seq_idx}] line_seq_ptr=0x{line_seq_ptr:04x}  repeat={line_count} lines")

        # Decode the unique lines in this sequence by following next_ptr links,
        # stopping when we see a self-loop or have walked enough distinct lines.
        # We decode each unique line once, then report the repeat count separately.
        unique_lines = []   # list of (line_ptr, entries, next_ptr)
        line_ptr = line_seq_ptr
        visited_in_seq = set()
        seq_safety = 0
        while True:
            if line_ptr in visited_in_seq:
                break  # hit a cycle — done collecting unique lines
            if line_ptr == 0xFFFF or line_ptr >= len(ram):
                break
            visited_in_seq.add(line_ptr)
            entries, next_ptr, _ = decode_line(ram, line_ptr, verbose)
            unique_lines.append((line_ptr, entries, next_ptr))
            if next_ptr == line_ptr:
                break  # self-loop
            line_ptr = next_ptr
            seq_safety += 1
            if seq_safety > 200:
                print("    WARNING: sequence walk safety limit hit")
                break

        n_unique = len(unique_lines)

        # Print each unique line once, with timing analysis.
        for ui, (lp, entries, np) in enumerate(unique_lines):
            total_px = sum(d for (_, d, _, _, _) in entries)
            print(f"    Line @0x{lp:04x}: {len(entries)} runs, total {total_px} px, next=0x{np:04x}")
            if verbose:
                print_line(entries, indent="      ")
            else:
                compact = []
                for (px, dur, sa, sb, sc) in entries:
                    a = " ".join(active_signals(sa, STATE_A_NAMES)) or "-"
                    c_cblank = "CBLANK_XMAP" if not (sc & 0x40) else ""
                    compact.append(f"@{px}+{dur}px [{a}]{' C:'+c_cblank if c_cblank else ''}")
                print(f"      " + "  ".join(compact))
                _analyze_line_timing(entries)

        # Report how many times the sequence cycles to produce line_count lines.
        if n_unique > 0:
            full_cycles = line_count // n_unique
            remainder   = line_count  % n_unique
            if n_unique == 1:
                print(f"    (line repeats {line_count}x total)")
            else:
                print(f"    ({n_unique} unique lines, {full_cycles} full cycles"
                      + (f" + {remainder} extra" if remainder else "") + f" = {line_count} lines total)")

        total_lines += line_count
        curr += 2
        seq_idx += 1
        safety += 1
        if safety > 200:
            print("  WARNING: frame table safety limit hit")
            break

    print(f"\n  Total lines in frame: {total_lines}")

def _analyze_line_timing(entries):
    """Print key signal transitions with pixel offsets for timing analysis."""
    key_events = []
    for (px, dur, sa, sb, sc) in entries:
        active_a = active_signals(sa, STATE_A_NAMES)
        active_c = active_signals(sc, STATE_C_NAMES)
        if active_a or ("CBLANK_XMAP" in active_c):
            key_events.append((px, dur, sa, sb, sc))

    if not key_events:
        return

    # Find key pixel offsets
    ser_en_start = None
    dsply_en_start = None
    vis_ln_start = None
    cblank_xmap_start = None
    hpos_start = None

    for (px, dur, sa, sb, sc) in entries:
        if not (sa & 0x08) and ser_en_start is None:      ser_en_start = px     # SER_EN active
        if not (sa & 0x04) and dsply_en_start is None:    dsply_en_start = px   # DSPLY_EN active
        if not (sa & 0x01) and vis_ln_start is None:      vis_ln_start = px     # VIS_LN active
        if not (sc & 0x40) and cblank_xmap_start is None: cblank_xmap_start = px # CBLANK_XMAP active
        if not (sa & 0x02) and hpos_start is None:        hpos_start = px       # HPOS active

    parts = []
    if hpos_start is not None:          parts.append(f"HPOS@{hpos_start}")
    if ser_en_start is not None:        parts.append(f"SER_EN@{ser_en_start}")
    if dsply_en_start is not None:      parts.append(f"DSPLY_EN@{dsply_en_start}")
    if vis_ln_start is not None:        parts.append(f"VIS_LN@{vis_ln_start}")
    if cblank_xmap_start is not None:   parts.append(f"CBLANK_XMAP@{cblank_xmap_start}")

    if len(parts) > 1:
        print(f"      --> Timing: {' | '.join(parts)}")
        # Show offsets between key signals
        if ser_en_start is not None and dsply_en_start is not None:
            delta = dsply_en_start - ser_en_start
            print(f"          SER_EN → DSPLY_EN: {delta:+d} px")
        if dsply_en_start is not None and vis_ln_start is not None:
            delta = vis_ln_start - dsply_en_start
            print(f"          DSPLY_EN → VIS_LN: {delta:+d} px")
        if ser_en_start is not None and vis_ln_start is not None:
            delta = vis_ln_start - ser_en_start
            print(f"          SER_EN → VIS_LN: {delta:+d} px  ← VRAM shifts {delta} px before vis area")
        if ser_en_start is not None and cblank_xmap_start is not None:
            delta = cblank_xmap_start - ser_en_start
            print(f"          SER_EN → CBLANK_XMAP: {delta:+d} px")

def decode_did(ram, did_ptr, scan_len, verbose=False):
    print(f"\n=== DID TABLE (did_ptr=0x{did_ptr:04x}, scan_len={scan_len}) ===")
    y = 0
    table_idx = did_ptr
    safety = 0
    while y < 1200 and table_idx < len(ram):
        line_ptr = ram[table_idx]
        if line_ptr == 0xFFFF:
            print(f"  y={y}: END marker (0xFFFF)")
            break
        ptr = line_ptr
        runs = []
        x = 0
        if ptr < len(ram):
            entry = ram[ptr]; ptr += 1
            current_did = entry & 0x1F
            while x < scan_len and ptr < len(ram):
                next_entry = ram[ptr]; ptr += 1
                next_x_raw = (next_entry >> 5) & 0x7FF
                next_did = next_entry & 0x1F
                is_eol = next_x_raw == 0x7FF
                next_x = scan_len if is_eol else next_x_raw
                run_end = min(max(next_x, x), scan_len)
                runs.append((x, run_end - x, current_did))
                x = run_end
                current_did = next_did
                if is_eol:
                    break
        if verbose or y < 10:
            run_str = "  ".join(f"x{r[0]}+{r[1]}:DID{r[2]}" for r in runs)
            print(f"  y={y:4d} @0x{line_ptr:04x}: {run_str}")
        elif y == 10:
            print(f"  ... (use -v to show all lines)")
        y += 1
        table_idx += 1
        safety += 1
        if safety > 2000:
            break

def load_dump(path):
    ram = {}
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # Parse metadata comments
                if '=' in line:
                    # e.g. "# VT_FRAME_PTR     = 0042"
                    parts = line.lstrip('#').strip().split('=')
                    key = parts[0].strip()
                    val_str = parts[1].strip().split()[0]
                    try:
                        meta[key] = int(val_str, 16)
                    except ValueError:
                        pass
                continue
            # Data line: "addr: w0 w1 w2 ..."
            colon = line.index(':')
            addr = int(line[:colon], 16)
            words = [int(w, 16) for w in line[colon+1:].split()]
            for i, w in enumerate(words):
                ram[addr + i] = w

    max_addr = max(ram.keys()) if ram else 0
    ram_list = [ram.get(i, 0) for i in range(max_addr + 1)]
    return ram_list, meta

def main():
    parser = argparse.ArgumentParser(description="Decode VC2 RAM dump (from 'vc2 ramdump')")
    parser.add_argument("dump_file", help="Path to vc2 ramdump output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all line runs")
    parser.add_argument("--vt-only", action="store_true", help="Only decode VT table")
    parser.add_argument("--did-only", action="store_true", help="Only decode DID table")
    parser.add_argument("--line", type=lambda x: int(x, 0), default=None,
                        help="Verbose decode of a specific line address (hex)")
    args = parser.parse_args()

    ram, meta = load_dump(args.dump_file)
    print(f"Loaded {len(ram)} words from {args.dump_file}")
    print(f"Metadata: {meta}")

    vt_ptr  = meta.get("VIDEO_ENTRY_PTR", meta.get("VT_FRAME_PTR", 0))
    did_ptr = meta.get("DID_ENTRY_PTR", 0)
    scan_len_raw = meta.get("SCANLINE_LEN", 0)
    scan_len = scan_len_raw >> 5

    if args.line is not None:
        entries, next_ptr, _ = decode_line(ram, args.line, verbose=True)
        total_px = sum(d for (_, d, _, _, _) in entries)
        print(f"\nLine @0x{args.line:04x}: {len(entries)} runs, {total_px} px, next=0x{next_ptr:04x}")
        print_line(entries, indent="  ")
        _analyze_line_timing(entries)
        return

    if not args.did_only:
        decode_vt(ram, vt_ptr, verbose=args.verbose)

    if not args.vt_only:
        decode_did(ram, did_ptr, scan_len if scan_len > 0 else 1280, verbose=args.verbose)

if __name__ == "__main__":
    main()
