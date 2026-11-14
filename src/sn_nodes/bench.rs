//! Micro-benchmarks of the paths a processor's access takes to its node's
//! registers and memory, on both SN families.
//!
//! Ignored, so `cargo test` skips them. Run with
//! `cargo test --release --lib bench_node_register_paths -- --ignored --nocapture`;
//! `BENCH_CASE` picks a case by name and `BENCH_N` sets the accesses per case
//! (default 20,000,000).
//!
//! Host instructions per access: run that command under
//! `perf stat -e instructions:u` with `BENCH_CASE` set and divide by `BENCH_N`.

use crate::dev::bedrock::regs::{lb, md, pi};
use crate::dev::bedrock::{Bedrock, Population};
use crate::dev::bedrock_flash::{FirmwareRecord, Flash};
use crate::dev::hub::{regs as hub_regs, Hub, HubConfig};
use crate::dev::heart::Clock;
use crate::dev::l1::{L1, L1Config};
use crate::ip35::node::NodeSpec;
use crate::sn_nodes::{NodeBoard, UncachedBus, UncachedSpace, IO_REGISTERS_LOCAL, IO_REGISTERS_NAMED, SN0_NODE_SHIFT, SN1_NODE_SHIFT};
use crate::traits::{BusDevice, BUS_OK};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

/// The processor's I/O space attribute in an uncached address.
const IO: u64 = 1 << 57;

#[test]
#[ignore]
fn bench_node_register_paths() {
    let n: u64 = std::env::var("BENCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000_000);
    let case = std::env::var("BENCH_CASE").unwrap_or_default();

    // SN0: one Hub with a 128 MiB bank, its processor bus wired as the
    // machine wires it.
    let hub = Hub::new(HubConfig { cpus: 2, banks_mb: [128, 0, 0, 0], ..Default::default() }, Arc::new(|| 0), None).unwrap();
    hub.decode.set_space(UncachedSpace::Io, hub.io_space());
    struct Nothing;
    impl BusDevice for Nothing {}
    let none: Arc<dyn BusDevice> = Arc::new(Nothing);
    let sn0 = UncachedBus::new([none.clone(), hub.processor_io(hub.cpu_view(UncachedSpace::Io)), none.clone(), hub.decode.physical().clone()]);
    let sn0: &dyn BusDevice = &sn0;
    let sn0_memory: &dyn BusDevice = &**hub.decode.physical();

    // SN0, two nodes: joined as the IP27 machine joins them, the second
    // numbered 1.
    let hub_node = |first_cpu| {
        let h = Hub::new(HubConfig { cpus: 2, first_cpu, banks_mb: [128, 0, 0, 0], nodes: 2, ..Default::default() }, Arc::new(|| 0), None).unwrap();
        h.decode.set_space(UncachedSpace::Io, h.io_space());
        let bus = UncachedBus::new([none.clone(), h.processor_io(h.cpu_view(UncachedSpace::Io)), none.clone(), h.decode.physical().clone()]);
        (h, bus)
    };
    let hubs = [hub_node(0), hub_node(2)];
    hubs[0].0.connect_peer(&hubs[1].0);
    hubs[1].0.connect_peer(&hubs[0].0);
    let hub_joined: &dyn BusDevice = &hubs[0].1;

    // SN1: one C-brick node, and two connected as the Origin 3200 connects
    // them, the second numbered 1 as its PROM numbers it.
    let spec = NodeSpec {
        name: "C-brick",
        module: &crate::cpu::parts::R12000_270_SYSCLK_90,
        firmware: FirmwareRecord::FUEL,
        board_id: 0,
        banks_mib: vec![512],
        processors: Population::SINGLE,
        absent: "",
        l1: L1Config::default(),
        flash: None,
    };
    let clock: Clock = Arc::new(|| 0);
    let (events, _) = std::sync::mpsc::channel();
    let build = |first_cpu, nodes| NodeBoard::<Bedrock, Flash>::build(&spec, &[], &clock, first_cpu, nodes, &events, &Arc::new(L1::new(L1Config::default()).unwrap())).unwrap();
    let single = build(0, 1);
    let sn1: &dyn BusDevice = &*single.bus;
    let sn1_memory: &dyn BusDevice = &**single.chip.decode.physical();
    let pair: Vec<_> = (0..2).map(|i| build(i, 2)).collect();
    pair[0].chip.connect_peer(&pair[1].chip);
    pair[1].chip.connect_peer(&pair[0].chip);
    let joined: &dyn BusDevice = &*pair[0].bus;
    let sn1_fetchop: &dyn BusDevice = &*single.bus;

    let hub_reg = IO | IO_REGISTERS_LOCAL | hub_regs::PI_REGION_PRESENT::OFFSET;
    let bedrock_reg = IO | IO_REGISTERS_LOCAL | pi::REGION_PRESENT::OFFSET;
    // The Bedrock answers a 4-byte read only at the real-time counter.
    let bedrock_counter = IO | IO_REGISTERS_LOCAL | (pi::RT_COUNTER::OFFSET + 4);
    let hub_peer_reg = IO | (1 << SN0_NODE_SHIFT) | IO_REGISTERS_NAMED | hub_regs::PI_REGION_PRESENT::OFFSET;
    let peer_reg = IO | (1 << SN1_NODE_SHIFT) | IO_REGISTERS_NAMED | pi::REGION_PRESENT::OFFSET;
    // The node's banks placed as its PROM places them, so a read at 1 MiB
    // reaches RAM.
    for bus in [sn1, &*pair[0].bus, &*pair[1].bus] {
        assert_eq!(bus.write64(IO | IO_REGISTERS_LOCAL | (md::BASE + md::MEMORY_CONFIG::OFFSET), 0x11), BUS_OK);
    }
    assert_eq!(hubs[1].0.write64(hub_regs::NI_STATUS_REV_ID::OFFSET, 1 << 8), BUS_OK);
    assert_eq!(pair[1].bus.write64(IO | IO_REGISTERS_LOCAL | (lb::BASE + lb::REV_ID::OFFSET), 1 << 32), BUS_OK);
    for (bus, a, a32) in [(sn0, hub_reg, hub_reg + 4), (sn1, bedrock_reg, bedrock_counter), (hub_joined, hub_reg, hub_reg + 4), (joined, bedrock_reg, bedrock_counter)] {
        assert_eq!(bus.read64(a).status, BUS_OK);
        assert_eq!(bus.read32(a32).status, BUS_OK);
        assert_eq!(bus.write64(a, bus.read64(a).data), BUS_OK);
    }
    for (bus, peer) in [(hub_joined, hub_peer_reg), (joined, peer_reg)] {
        assert_eq!(bus.read64(peer).status, BUS_OK, "node 1's registers by its number");
        assert_eq!(bus.write64(peer, bus.read64(peer).data), BUS_OK);
    }
    // Node 1's memory, read through node 0's table.
    let (sn0_peer, sn1_peer) = (1u64 << SN0_NODE_SHIFT, 1u64 << SN1_NODE_SHIFT);
    let hub2_memory: &dyn BusDevice = &**hubs[0].0.decode.physical();
    let sn1_pair_memory: &dyn BusDevice = &**pair[0].chip.decode.physical();
    for (memory, node1) in [(hub2_memory, sn0_peer), (sn1_pair_memory, sn1_peer)] {
        assert_eq!(memory.write64(node1 | 1 << 20, 0xa5a5), BUS_OK, "node1 {node1:#x}");
        assert_eq!(memory.read64(node1 | 1 << 20).data, 0xa5a5, "node 1's RAM at 1 MiB");
    }
    // Fetch-and-op in the memory-special space: attribute 2, operation 8 steps the word.
    let fetchop = (2u64 << 57) | (1 << 20) | 8;
    assert_eq!(sn1_fetchop.write64(1 << 20, 41), BUS_OK);
    assert_eq!(sn1_fetchop.read64(fetchop).data, 41);
    assert_eq!(sn1_memory.read64(1 << 20).data, 42, "the fetch-op stepped the word");
    for memory in [sn0_memory, sn1_memory] {
        assert_eq!(memory.write64(1 << 20, 0x5a5a), BUS_OK);
        assert_eq!(memory.read64(1 << 20).data, 0x5a5a, "RAM at 1 MiB");
    }

    // Lines 128 bytes apart over 512 KiB, so the reads walk host pages.
    let line = |i: u64| (1 << 20) + (i & 4095) * 128;
    let cases: [(&str, &dyn Fn(u64) -> u64); 22] = [
        ("loop", &|i| i),
        ("hub-read64", &|_| sn0.read64(black_box(hub_reg)).data),
        ("hub-write64", &|i| u64::from(sn0.write64(black_box(hub_reg), i))),
        ("hub-read32", &|_| u64::from(sn0.read32(black_box(hub_reg + 4)).data)),
        ("hub2-read64", &|_| hub_joined.read64(black_box(hub_reg)).data),
        ("hub2-write64", &|i| u64::from(hub_joined.write64(black_box(hub_reg), i))),
        ("hub2-read32", &|_| u64::from(hub_joined.read32(black_box(hub_reg + 4)).data)),
        ("hub2-peer-read64", &|_| hub_joined.read64(black_box(hub_peer_reg)).data),
        ("hub2-peer-write64", &|i| u64::from(hub_joined.write64(black_box(hub_peer_reg), i))),
        ("bedrock-read64", &|_| sn1.read64(black_box(bedrock_reg)).data),
        ("bedrock-write64", &|i| u64::from(sn1.write64(black_box(bedrock_reg), i))),
        ("bedrock-read32", &|_| u64::from(sn1.read32(black_box(bedrock_counter)).data)),
        ("bedrock2-read64", &|_| joined.read64(black_box(bedrock_reg)).data),
        ("bedrock2-write64", &|i| u64::from(joined.write64(black_box(bedrock_reg), i))),
        ("bedrock2-read32", &|_| u64::from(joined.read32(black_box(bedrock_counter)).data)),
        ("bedrock2-peer-read64", &|_| joined.read64(black_box(peer_reg)).data),
        ("bedrock2-peer-write64", &|i| u64::from(joined.write64(black_box(peer_reg), i))),
        ("sn0-memory-read64", &|i| sn0_memory.read64(black_box(line(i))).data),
        ("sn1-memory-read64", &|i| sn1_memory.read64(black_box(line(i))).data),
        ("sn0-peer-memory-read64", &|i| hub2_memory.read64(black_box(sn0_peer | line(i))).data),
        ("sn1-peer-memory-read64", &|i| sn1_pair_memory.read64(black_box(sn1_peer | line(i))).data),
        ("sn1-fetchop-read64", &|_| sn1_fetchop.read64(black_box(fetchop)).data),
    ];
    for (name, f) in cases {
        if case.is_empty() || case == name {
            let t = Instant::now();
            let mut sum = 0u64;
            for i in 0..n {
                sum = sum.wrapping_add(f(black_box(i)));
            }
            black_box(sum);
            println!("{name:22} {:7.2} ns", t.elapsed().as_nanos() as f64 / n as f64);
        }
    }

    // A node's table points into its peer's memory: release every node's
    // devices before any goes.
    for memory in [hub.decode.physical(), hubs[0].0.decode.physical(), hubs[1].0.decode.physical(), single.chip.decode.physical(), pair[0].chip.decode.physical(), pair[1].chip.decode.physical()] {
        memory.release_devices();
    }
}
