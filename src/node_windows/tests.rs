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
