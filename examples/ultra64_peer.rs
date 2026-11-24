//! A stand-in for the N64 side of the Ultra64 bridge, for testing the Indy
//! board without gopher64. It opens the running iris's shared memory, queues
//! the N64-to-Indy messages named on the command line and signals them, as
//! gopher64 does.
//!
//! ```text
//! cargo run --example ultra64-peer -- gio 2a rdb-write 1234 rdb-read sync 7
//! ```
//!
//! `gio <hex>` raises the GIO interrupt with a payload, `rdb-write <hex>`
//! delivers an RDB packet (PROD_ID bit 31), `rdb-read` reports that the N64
//! read the Indy's packet (bit 30), and `sync <hex>` sets GIO_SYNC. With
//! `--status`, it prints both rings' head and tail instead.
//!
//! `tools/tests/ultra64-irq-or.py` drives it against a booted Indy.

use iris::ultra_proto::{n2h, ShmHeader, SHM_HEADER_OFFSET, SHM_MAGIC};
use raw_sync::events::{Event, EventInit, EventState};
use shared_memory::ShmemConf;

const SHM_OS_ID: &str = "iris_n64_bridge";

fn hex(arg: Option<&String>, what: &str) -> u32 {
    let s = arg.unwrap_or_else(|| die(&format!("{what} needs a hex value")));
    u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or_else(|_| die(&format!("bad hex {s:?}")))
}

fn die(msg: &str) -> ! {
    eprintln!("ultra64-peer: {msg}");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let shm = ShmemConf::new().os_id(SHM_OS_ID).open()
        .unwrap_or_else(|e| die(&format!("cannot open the bridge ({e}); is iris running with an ultra64 device?")));
    let base = shm.as_ptr();
    // iris lays out the h2n event first and the n2h event right after it.
    let (_h2n, used) = unsafe { Event::from_existing(base) }
        .unwrap_or_else(|e| die(&format!("h2n event: {e}")));
    let (n2h_evt, _) = unsafe { Event::from_existing(base.add(used)) }
        .unwrap_or_else(|e| die(&format!("n2h event: {e}")));
    let hdr = unsafe { &mut *(base.add(SHM_HEADER_OFFSET) as *mut ShmHeader) };
    if hdr.magic != SHM_MAGIC {
        die(&format!("bad bridge magic {:#010x}", hdr.magic));
    }

    if args.first().map(String::as_str) == Some("--status") {
        println!("h2n {}/{} n2h {}/{}", hdr.h2n_ring.head, hdr.h2n_ring.tail, hdr.n2h_ring.head, hdr.n2h_ring.tail);
        return;
    }

    let mut it = args.iter();
    let mut sent = 0;
    while let Some(cmd) = it.next() {
        let (kind, payload) = match cmd.as_str() {
            "gio" => (n2h::GIO_INT, hex(it.next(), "gio")),
            "rdb-write" => (n2h::RDB_WRITE, hex(it.next(), "rdb-write")),
            "rdb-read" => (n2h::RDB_READ, 0),
            "sync" => (n2h::GIO_SYNC, hex(it.next(), "sync")),
            other => die(&format!("unknown message {other:?} (gio, rdb-write, rdb-read, sync)")),
        };
        if !hdr.n2h_ring.push(kind, payload) {
            die("n2h ring full");
        }
        sent += 1;
    }
    if sent == 0 {
        die("usage: ultra64-peer [--status] | <gio HEX | rdb-write HEX | rdb-read | sync HEX>...");
    }
    n2h_evt.set(EventState::Signaled).unwrap_or_else(|e| die(&format!("signal: {e}")));
    println!("sent {sent}");
}
