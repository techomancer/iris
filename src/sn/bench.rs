//! Micro-benchmarks of a processor's access paths to its node's registers and memory, on both
//! IP27 and IP35 machines. Ignored; run `cargo test --release --lib bench_node_register_paths --
//! --ignored --nocapture`, `BENCH_CASE` picks a case, `BENCH_N` sets accesses (default
//! 20,000,000); under `perf stat -e instructions:u`, divide by `BENCH_N`.

use crate::dev::bedrock::regs::{lb, md, pi};
use crate::dev::bedrock::{self, Bedrock, Population};
use crate::dev::bedrock_flash::{FirmwareRecord, Flash};
use crate::dev::hub::{self, regs as hub_regs, Hub, HubConfig};
use crate::dev::node_flash::{FlashValues, NodeFlash};
use crate::dev::heart::Clock;
use crate::dev::l1::{L1, L1Config};
use crate::ip35::node::NodeSpec;
use crate::sn::{NodeBoard, IO_REGISTERS_LOCAL, IO_REGISTERS_NAMED};
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

    // IP27: one Hub node with a 128 MiB bank, and two joined as the IP27 machine joins them, the
    // second numbered 1.
    let clock: Clock = Arc::new(|| 0);
    let prom = super::tests::prom();
    let hub_spec = |index: usize, nodes| crate::ip27::NodeSpec {
        hub: HubConfig { cpus: 1, first_cpu: 2 * index, banks_mb: [128, 0, 0, 0], nodes, ..Default::default() },
        module: crate::cpu::parts::cpu_module(&crate::cpu::parts::R10000_180).unwrap(),
        firmware: FlashValues::STOCK_R10000_180,
        flash: None,
        node_record: None,
    };
    let hub_node = |index, nodes| NodeBoard::<Hub, NodeFlash>::build("origin2000", &hub_spec(index, nodes), index, &prom, &clock).unwrap();
    let hub = hub_node(0, 1);
    let hub_bus: &dyn BusDevice = &*hub.bus;
    let hub_memory: &dyn BusDevice = &**hub.chip.decode.physical();
    let hubs = [hub_node(0, 2), hub_node(1, 2)];
    hubs[0].chip.connect_peer(&hubs[1].chip);
    hubs[1].chip.connect_peer(&hubs[0].chip);
    let hub_joined: &dyn BusDevice = &*hubs[0].bus;

    // IP35: one C-brick node, and two connected as the Origin 3200 connects
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
    let (events, _) = std::sync::mpsc::channel();
    let build = |first_cpu, nodes| {
        NodeBoard::<Bedrock, Flash>::build(
            "fuel",
            &spec,
            &[],
            &clock,
            first_cpu,
            nodes,
            &events,
            &Arc::new(L1::new(L1Config::default()).unwrap()),
        )
        .unwrap()
    };
    let single = build(0, 1);
    let bedrock_bus: &dyn BusDevice = &*single.bus;
    let bedrock_memory: &dyn BusDevice = &**single.chip.decode.physical();
    let pair: Vec<_> = (0..2).map(|i| build(i, 2)).collect();
    pair[0].chip.connect_peer(&pair[1].chip);
    pair[1].chip.connect_peer(&pair[0].chip);
    let joined: &dyn BusDevice = &*pair[0].bus;
    let bedrock_fetchop: &dyn BusDevice = &*single.bus;

    let hub_reg = IO | IO_REGISTERS_LOCAL | hub_regs::PI_REGION_PRESENT::OFFSET;
    let bedrock_reg = IO | IO_REGISTERS_LOCAL | pi::REGION_PRESENT::OFFSET;
    // The Bedrock answers a 4-byte read only at the real-time counter.
    let bedrock_counter = IO | IO_REGISTERS_LOCAL | (pi::RT_COUNTER::OFFSET + 4);
    let hub_peer_reg = IO | (1 << hub::NODE_SHIFT) | IO_REGISTERS_NAMED | hub_regs::PI_REGION_PRESENT::OFFSET;
    let peer_reg = IO | (1 << bedrock::NODE_SHIFT) | IO_REGISTERS_NAMED | pi::REGION_PRESENT::OFFSET;
    // The node's banks placed as its PROM places them, so a read at 1 MiB
    // reaches RAM.
    for bus in [bedrock_bus, &*pair[0].bus, &*pair[1].bus] {
        assert_eq!(bus.write64(IO | IO_REGISTERS_LOCAL | (md::BASE + md::MEMORY_CONFIG::OFFSET), 0x11), BUS_OK);
    }
    assert_eq!(hubs[1].chip.write64(hub_regs::NI_STATUS_REV_ID::OFFSET, 1 << 8), BUS_OK);
    assert_eq!(pair[1].bus.write64(IO | IO_REGISTERS_LOCAL | (lb::BASE + lb::REV_ID::OFFSET), 1 << 32), BUS_OK);
    for (bus, a, a32) in [
        (hub_bus, hub_reg, hub_reg + 4),
        (bedrock_bus, bedrock_reg, bedrock_counter),
        (hub_joined, hub_reg, hub_reg + 4),
        (joined, bedrock_reg, bedrock_counter),
    ] {
        assert_eq!(bus.read64(a).status, BUS_OK);
        assert_eq!(bus.read32(a32).status, BUS_OK);
        assert_eq!(bus.write64(a, bus.read64(a).data), BUS_OK);
    }
    for (bus, peer) in [(hub_joined, hub_peer_reg), (joined, peer_reg)] {
        assert_eq!(bus.read64(peer).status, BUS_OK, "node 1's registers by its number");
        assert_eq!(bus.write64(peer, bus.read64(peer).data), BUS_OK);
    }
    // Node 1's memory, read through node 0's table.
    let (hub_peer, bedrock_peer) = (1u64 << hub::NODE_SHIFT, 1u64 << bedrock::NODE_SHIFT);
    let hub_pair_memory: &dyn BusDevice = &**hubs[0].chip.decode.physical();
    let bedrock_pair_memory: &dyn BusDevice = &**pair[0].chip.decode.physical();
    for (memory, node1) in [(hub_pair_memory, hub_peer), (bedrock_pair_memory, bedrock_peer)] {
        assert_eq!(memory.write64(node1 | 1 << 20, 0xa5a5), BUS_OK, "node1 {node1:#x}");
        assert_eq!(memory.read64(node1 | 1 << 20).data, 0xa5a5, "node 1's RAM at 1 MiB");
    }
    // Fetch-and-op in the memory-special space: attribute 2, operation 8 steps the word.
    let fetchop = (2u64 << 57) | (1 << 20) | 8;
    assert_eq!(bedrock_fetchop.write64(1 << 20, 41), BUS_OK);
    assert_eq!(bedrock_fetchop.read64(fetchop).data, 41);
    assert_eq!(bedrock_memory.read64(1 << 20).data, 42, "the fetch-op stepped the word");
    for memory in [hub_memory, bedrock_memory] {
        assert_eq!(memory.write64(1 << 20, 0x5a5a), BUS_OK);
        assert_eq!(memory.read64(1 << 20).data, 0x5a5a, "RAM at 1 MiB");
    }

    // Lines 128 bytes apart over 512 KiB, so the reads walk host pages.
    let line = |i: u64| (1 << 20) + (i & 4095) * 128;
    let cases: [(&str, &dyn Fn(u64) -> u64); 22] = [
        ("loop", &|i| i),
        ("hub-read64", &|_| hub_bus.read64(black_box(hub_reg)).data),
        ("hub-write64", &|i| u64::from(hub_bus.write64(black_box(hub_reg), i))),
        ("hub-read32", &|_| u64::from(hub_bus.read32(black_box(hub_reg + 4)).data)),
        ("hub-2node-read64", &|_| hub_joined.read64(black_box(hub_reg)).data),
        ("hub-2node-write64", &|i| u64::from(hub_joined.write64(black_box(hub_reg), i))),
        ("hub-2node-read32", &|_| u64::from(hub_joined.read32(black_box(hub_reg + 4)).data)),
        ("hub-peer-read64", &|_| hub_joined.read64(black_box(hub_peer_reg)).data),
        ("hub-peer-write64", &|i| u64::from(hub_joined.write64(black_box(hub_peer_reg), i))),
        ("bedrock-read64", &|_| bedrock_bus.read64(black_box(bedrock_reg)).data),
        ("bedrock-write64", &|i| u64::from(bedrock_bus.write64(black_box(bedrock_reg), i))),
        ("bedrock-read32", &|_| u64::from(bedrock_bus.read32(black_box(bedrock_counter)).data)),
        ("bedrock-2node-read64", &|_| joined.read64(black_box(bedrock_reg)).data),
        ("bedrock-2node-write64", &|i| u64::from(joined.write64(black_box(bedrock_reg), i))),
        ("bedrock-2node-read32", &|_| u64::from(joined.read32(black_box(bedrock_counter)).data)),
        ("bedrock-peer-read64", &|_| joined.read64(black_box(peer_reg)).data),
        ("bedrock-peer-write64", &|i| u64::from(joined.write64(black_box(peer_reg), i))),
        ("hub-memory-read64", &|i| hub_memory.read64(black_box(line(i))).data),
        ("bedrock-memory-read64", &|i| bedrock_memory.read64(black_box(line(i))).data),
        ("hub-peer-memory-read64", &|i| hub_pair_memory.read64(black_box(hub_peer | line(i))).data),
        ("bedrock-peer-memory-read64", &|i| bedrock_pair_memory.read64(black_box(bedrock_peer | line(i))).data),
        ("bedrock-fetchop-read64", &|_| bedrock_fetchop.read64(black_box(fetchop)).data),
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
    for memory in [
        hub.chip.decode.physical(),
        hubs[0].chip.decode.physical(),
        hubs[1].chip.decode.physical(),
        single.chip.decode.physical(),
        pair[0].chip.decode.physical(),
        pair[1].chip.decode.physical(),
    ] {
        memory.release_devices();
    }
}
