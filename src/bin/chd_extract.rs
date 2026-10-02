//! One-shot tool: extract a SCSI HD CHD to a raw disk image.
//!
//! Usage: chd_extract <input.chd> <output.raw>

use std::env;
use std::io::Write;
use std::process::exit;
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <input.chd> <output.raw>", args[0]);
        exit(2);
    }
    let input  = &args[1];
    let output = &args[2];

    // Extract reads through the merged view; COW off (no diff is created here).
    let mut chd = match iris::chd_disk::ChdHd::open(input, false) {
        Ok(c) => c,
        Err(e) => { eprintln!("open {}: {}", input, e); exit(1); }
    };
    let total = chd.size();
    let sector_size: u64 = 512;
    let total_sectors = total / sector_size;
    eprintln!("CHD: {} bytes ({} sectors of {} bytes)", total, total_sectors, sector_size);

    let mut out = match std::fs::File::create(output) {
        Ok(f) => f,
        Err(e) => { eprintln!("create {}: {}", output, e); exit(1); }
    };

    // Read in chunks of 1024 sectors (512 KB) to keep memory use modest.
    const CHUNK: u64 = 1024;
    let mut lba: u64 = 0;
    let mut last_pct: u64 = 0;
    while lba < total_sectors {
        let n = std::cmp::min(CHUNK, total_sectors - lba);
        let buf = match chd.read_blocks(lba, n as usize, sector_size) {
            Ok(b) => b,
            Err(e) => { eprintln!("read at lba {}: {}", lba, e); exit(1); }
        };
        if let Err(e) = out.write_all(&buf) {
            eprintln!("write at lba {}: {}", lba, e); exit(1);
        }
        lba += n;
        let pct = (lba * 100) / total_sectors;
        if pct >= last_pct + 5 {
            eprintln!("  {}% ({}/{} sectors)", pct, lba, total_sectors);
            last_pct = pct;
        }
    }
    eprintln!("done: wrote {} bytes to {}", total, output);
}

