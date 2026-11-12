use super::*;
use crate::ip27::{NodeMemory, NodeZero};
use crate::ppmem::{bitmap_maps_at, window_shift, MappedMemory, FINE_SHIFT};
use std::sync::atomic::Ordering;

/// Stands in for the Hub's node-qualified memory, whose presence tells a node
/// its other node numbers are routed.
struct Echo;
impl BusDevice for Echo {}

/// Two SN1 nodes' windows follow the numbers their Hubs publish with the node
/// number at address bit 33: each shows its own memory and the other's at
/// `number << 33`, nothing at the SN0 position, and a node holding another
/// number keeps its own memory below the cached alias and shows node zero's
/// above it. Memory at 2 MiB multiples is what the jitv2 build maps.
#[test]
fn sn1_windows_show_both_nodes_memory_at_bit_33() {
    let windows = Arc::new(Windows::new(SN1_NODE_SHIFT));
    let echo: Arc<dyn BusDevice> = Arc::new(Echo);
    // Room for node numbers 0 to 3, so number two fits.
    let size = 4u64 << SN1_NODE_SHIFT;
    let mems: Vec<_> = (0..2).map(|_| NodeMemory::with_window([128, 0, 0, 0], size, window_shift(size)).unwrap()).collect();
    let routing: Vec<_> = (0..2).map(|_| Routing::new()).collect();
    let zeros: Vec<Arc<NodeZero>> = mems
        .iter()
        .enumerate()
        .map(|(n, m)| {
            m.route_nodes(echo.clone());
            m.node_zero(routing[n].clone(), &windows)
        })
        .collect();
    let (near, mid, far) = (0x1000, (2 << 20) + 0x1000, (96 << 20) + 0x1000);
    for (n, m) in mems.iter().enumerate() {
        let n = n as u64;
        assert_eq!(m.physical().banks()[0].write64(near, n << 4), BUS_OK);
        assert_eq!(m.physical().banks()[0].write64(far, n << 4 | 1), BUS_OK);
        assert_eq!(m.physical().banks()[0].write64(mid, n << 4 | 2), BUS_OK);
    }
    let seen = |n: usize, at: u64| {
        let sp = mems[n].physical().ppmem_space().unwrap();
        let (fine, regions) = sp.fine_table();
        let r = at >> FINE_SHIFT;
        let whole = r < regions && unsafe { (*fine.add((r / 64) as usize)).load(Ordering::Relaxed) } >> (r % 64) & 1 != 0;
        // The fine table holds everything the bitmap does.
        assert!(whole || !bitmap_maps_at(sp.mapped_bitmap(), sp.region_shift(), at));
        whole.then(|| unsafe { (*(sp.window_base().add(at as usize) as *const u64)).rotate_left(32) })
    };
    let alias = 64 << 20;
    let (sn0, sn1) = (1u64 << 32, 1u64 << 33);
    // Both hold zero, the reset value: each window is its own memory.
    assert_eq!((seen(0, near), seen(1, near)), (Some(0x00), Some(0x10)));
    assert_eq!((seen(0, sn1 + near), seen(1, sn1 + near)), (None, None));
    // The second holds one: its memory is at bit 33, not bit 32.
    zeros[1].update(1, alias);
    assert_eq!(seen(0, near), Some(0x00));
    assert_eq!(seen(0, sn1 + far), Some(0x11));
    assert_eq!(seen(0, sn0 + far), None);
    assert_eq!(seen(1, near), Some(0x10));
    assert_eq!(seen(1, far), Some(0x01));
    assert_eq!(seen(1, sn1 + near), Some(0x10));
    assert_eq!(seen(1, sn0 + near), None);
    // A smaller alias shows more of node zero to the second node.
    zeros[1].update(1, 2 << 20);
    assert_eq!(seen(1, near), Some(0x10));
    assert_eq!(seen(1, mid), Some(0x02));
    // Two: one leaves both windows, and the second node's memory is at two.
    zeros[1].update(2, alias);
    assert_eq!((seen(0, sn1 + near), seen(1, sn1 + near)), (None, None));
    assert_eq!((seen(0, (2 << 33) + near), seen(1, (2 << 33) + far)), (Some(0x10), Some(0x11)));
    assert_eq!(seen(1, near), Some(0x10));
    // Back to zero.
    zeros[1].update(0, alias);
    assert_eq!((seen(0, near), seen(1, near), seen(1, far)), (Some(0x00), Some(0x10), Some(0x11)));
}

/// A piece the host cannot map leaves the window it was to replace. The old
/// piece is no longer what the Hubs route and nothing records it, so it must
/// not stay mapped showing the wrong node's memory. The processor reaches the
/// span over the bus instead, and the next layout tries the piece again.
#[test]
fn a_piece_the_host_cannot_map_leaves_the_window_and_is_tried_again() {
    let (size, shift) = crate::ip27::window(4);
    let windows = Arc::new(Windows::default());
    let echo: Arc<dyn BusDevice> = Arc::new(Echo);
    let mems: Vec<_> = (0..2).map(|_| NodeMemory::with_window([128, 0, 0, 0], size, shift).unwrap()).collect();
    let routing: Vec<_> = (0..2).map(|_| Routing::new()).collect();
    let _zeros: Vec<_> = mems
        .iter()
        .enumerate()
        .map(|(n, m)| {
            m.route_nodes(echo.clone());
            m.node_zero(routing[n].clone(), &windows)
        })
        .collect();
    let far = (96 << 20) + 0x1000;
    for (n, m) in mems.iter().enumerate() {
        assert_eq!(m.physical().banks()[0].write64(far, n as u64 + 1), BUS_OK);
    }
    let seen = |n: usize, at: u64| {
        let sp = mems[n].physical().ppmem_space().unwrap();
        let (fine, regions) = sp.fine_table();
        let r = at >> FINE_SHIFT;
        let whole = r < regions && unsafe { (*fine.add((r / 64) as usize)).load(Ordering::Relaxed) } >> (r % 64) & 1 != 0;
        assert!(whole || !bitmap_maps_at(sp.mapped_bitmap(), sp.region_shift(), at));
        whole.then(|| unsafe { (*(sp.window_base().add(at as usize) as *const u64)).rotate_left(32) })
    };
    // Both hold zero: node one's window is its own memory.
    assert_eq!(seen(1, far), Some(2));
    // Node one takes number one, and the host refuses the node-zero piece it
    // would show above the alias.
    routing[1].set(1, 64 << 20);
    {
        let mut nodes = windows.nodes.lock();
        let mut refused = 0;
        Windows::lay_by(&mut nodes, 1, SN0_NODE_SHIFT, &mut |sp, bank, p| {
            if p.owner == 0 {
                refused += 1;
                return Err(std::io::Error::other("injected mapping failure"));
            }
            sp.map_range_of(bank, p.src, p.at, p.len)
        });
        assert_eq!(refused, 1, "one piece, the node-zero memory above the alias");
        assert!(nodes[1].laid.iter().all(|p| p.owner == 1), "the refused piece is not recorded");
    }
    assert_eq!(seen(1, far), None, "the window kept the old node's memory at an address it no longer shows");
    assert!(seen(1, 0x1000).is_some(), "memory below the alias is untouched");
    // The next layout does it again.
    windows.refresh();
    assert_eq!(seen(1, far), Some(1), "node zero's memory, now that the host maps it");
}

/// Two SN0 nodes of 128 MB, both still number zero, on one board.
#[cfg(feature = "jitv2")]
fn two_sn0_nodes() -> (Vec<NodeMemory>, Vec<Arc<NodeZero>>) {
    let (size, shift) = crate::ip27::window(4);
    let windows = Arc::new(Windows::default());
    let echo: Arc<dyn BusDevice> = Arc::new(Echo);
    let mems: Vec<_> = (0..2).map(|_| NodeMemory::with_window([128, 0, 0, 0], size, shift).unwrap()).collect();
    let zeros = mems
        .iter()
        .map(|m| {
            m.route_nodes(echo.clone());
            m.node_zero(Routing::new(), &windows)
        })
        .collect();
    (mems, zeros)
}

/// The generation counter of `at` in node `m`'s own bank.
#[cfg(feature = "jitv2")]
fn bank_counter(m: &NodeMemory, at: u64) -> &std::sync::atomic::AtomicU64 {
    unsafe { &*m.physical().banks()[0].gen_ptr(at) }
}

/// What a processor's CACHE instruction does to a page's counter when its
/// cache takes the page's first line and lets go of its last
/// (`PpMemSpace::mark`).
#[cfg(feature = "jitv2")]
fn hold(c: &std::sync::atomic::AtomicU64) {
    c.fetch_add(crate::cpu::jitv2::jitv2::GEN_HELD, Ordering::Release);
}
#[cfg(feature = "jitv2")]
fn release(c: &std::sync::atomic::AtomicU64) {
    c.fetch_add(1u64.wrapping_sub(crate::cpu::jitv2::jitv2::GEN_HELD), Ordering::Release);
}

/// Above node one's cached alias once it takes number one.
#[cfg(feature = "jitv2")]
const FAR: u64 = (96 << 20) + 0x1000;
#[cfg(feature = "jitv2")]
const ALIAS: u64 = 64 << 20;

/// A page that took 2^32 writes between layouts must not let a layout raise
/// another bank's counter to a value the busy page handed out, as adding
/// `layout << 32` to every counter did.
#[cfg(feature = "jitv2")]
#[test]
fn a_renumbering_raises_counters_past_a_page_written_2_32_times() {
    let (mems, zeros) = two_sn0_nodes();
    // Node zero's own page was written past 2^32 times, and the JIT claimed it.
    let busy = bank_counter(&mems[0], FAR);
    busy.store((3 << 32) - 1, Ordering::Release);
    assert_eq!(mems[0].physical().write64(FAR, 1), BUS_OK);
    assert_eq!(mems[0].physical().write64(FAR, 2), BUS_OK);
    let held = busy.load(Ordering::Acquire);
    // Node zero takes number one: above its alias its window shows node one.
    zeros[0].update(1, ALIAS);
    let claimed = mems[0].physical().gen_ptr(FAR);
    let gen = || unsafe { (*claimed).load(Ordering::Acquire) };
    assert!(gen() > held, "{:#x} should be past {held:#x}", gen());
    // A write to node one's page must not bring the counter back to the
    // value the page claimed from node zero.
    assert_eq!(mems[1].physical().write64(FAR, 3), BUS_OK);
    assert!(gen() > held);
}

/// A page written close to 2^48 times elsewhere must not lift the pages a
/// layout lays into the hold bits, where the JIT would take them as held by a
/// cache and never compile them.
#[cfg(feature = "jitv2")]
#[test]
fn a_page_written_near_the_top_of_its_count_leaves_laid_pages_compilable() {
    use crate::cpu::jitv2::jitv2::GEN_HELD;
    let (mems, zeros) = two_sn0_nodes();
    let busy = (0xffff << 32) + 5;
    bank_counter(&mems[0], 0x1000).store(busy, Ordering::Release);
    let claimed = mems[1].physical().gen_ptr(FAR);
    let gen = || unsafe { (*claimed).load(Ordering::Acquire) };
    zeros[1].update(1, ALIAS);
    assert!(gen() > busy, "{:#x} should be past {busy:#x}", gen());
    assert!(gen() < GEN_HELD, "{:#x} reads as held, and nothing holds it", gen());
}

/// A page a processor's cache holds elsewhere carries `GEN_HELD`. The layout
/// must look past the hold to the writes, or it raises every page it lays
/// into the hold bits, where they read as held for good.
#[cfg(feature = "jitv2")]
#[test]
fn a_page_held_elsewhere_does_not_leave_laid_pages_held() {
    use crate::cpu::jitv2::jitv2::GEN_HELD;
    let (mems, zeros) = two_sn0_nodes();
    // Node zero's processor holds a line of a page below the alias, which
    // this layout does not lay anywhere new.
    let elsewhere = bank_counter(&mems[0], 0x1000);
    hold(elsewhere);
    let claimed = mems[1].physical().gen_ptr(FAR);
    let gen = || unsafe { (*claimed).load(Ordering::Acquire) };
    let before = gen();
    zeros[1].update(1, ALIAS);
    let after = gen();
    assert!(after > before, "the laid page reads {after:#x}, not past {before:#x}, the value node one's JIT claimed from its own memory");
    assert!(after < GEN_HELD, "{after:#x} reads as held, and nothing holds it");
    // The hold comes off the page that had it, as it went on.
    assert_eq!(elsewhere.load(Ordering::Acquire), GEN_HELD);
    release(elsewhere);
    assert_eq!(elsewhere.load(Ordering::Acquire), 1);
}

/// A laid page whose counter carries a hold must be raised under the hold, in
/// its write count. Left as it is because the hold bits put it above the
/// floor, it drops back when the cache lets go, onto values the page claimed
/// from the memory it showed before.
#[cfg(feature = "jitv2")]
#[test]
fn a_held_page_laid_and_let_go_does_not_fall_back_to_a_value_claimed_before() {
    use crate::cpu::jitv2::jitv2::GEN_HELD;
    let (mems, zeros) = two_sn0_nodes();
    // The JIT on node one claimed its own page after 2^32 + 1 writes.
    bank_counter(&mems[1], FAR).store((1 << 32) + 1, Ordering::Release);
    let claimed = mems[1].physical().gen_ptr(FAR);
    let gen = || unsafe { (*claimed).load(Ordering::Acquire) };
    let before = gen();
    // Node zero's processor holds a line of node zero's page there.
    let laid = bank_counter(&mems[0], FAR);
    hold(laid);
    zeros[1].update(1, ALIAS);
    assert!(gen() >= GEN_HELD, "{:#x}: the page is held, so the JIT compiles nothing there", gen());
    release(laid);
    let after = gen();
    assert!(after < GEN_HELD, "{after:#x}: let go, the page compiles again");
    assert!(after > before, "{after:#x} fell back to {before:#x} or below");
}

/// A layout once some page has been written 2^48 - 1 times has no write count
/// left to raise to. The pages it lays are retired instead: they read as held
/// from then on, so the JIT never compiles them, and a hold put on before
/// still comes off as it went on.
#[cfg(feature = "jitv2")]
#[test]
fn a_layout_past_the_top_of_the_count_retires_the_pages_it_lays() {
    use crate::cpu::jitv2::jitv2::GEN_HELD;
    let (mems, zeros) = two_sn0_nodes();
    bank_counter(&mems[0], 0x1000).store(GEN_HELD - 1, Ordering::Release);
    let laid = bank_counter(&mems[0], FAR);
    hold(laid);
    let claimed = mems[1].physical().gen_ptr(FAR);
    let gen = || unsafe { (*claimed).load(Ordering::Acquire) };
    zeros[1].update(1, ALIAS);
    assert!(gen() >= GEN_HELD);
    release(laid);
    assert!(gen() >= GEN_HELD, "{:#x}: a retired page compiles again", gen());
    assert_eq!(mems[0].physical().banks()[0].write64(FAR, 1), BUS_OK);
    assert!(gen() >= GEN_HELD, "{:#x}: a write let a retired page compile", gen());
    // Node one's own page is laid in both windows, at number one, and takes
    // one hold for it, not one for each.
    assert_eq!(bank_counter(&mems[1], FAR).load(Ordering::Acquire), GEN_HELD);
    // The page that ran out was not laid, and is left as it was.
    assert_eq!(bank_counter(&mems[0], 0x1000).load(Ordering::Acquire), GEN_HELD - 1);
}
