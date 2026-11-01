//! The Octane's chips wired together as `ip30_machine` wires them, driven
//! without a CPU.
//!
//! The first group is the processor's bus: `Ip30Bus` over HEART, the
//! Crossbow, the Bridge with the boot flash, SDRAM and its first 16 KiB at physical 0.
//!
//! The rest are the real HEART, Bridge, IOC3, IOC3 Ethernet and ISP1040,
//! programmed the way the PROM and the guest program them, to catch
//! byte-order and wiring mistakes that each device's own tests cannot see.
//! The processor's accesses go straight to the Bridge's small window
//! (offsets as in physical 0x1f00_0000 + offset); a function's DMA goes
//! through the Bridge into memory at physical 0x2000_0000. Expected bytes
//! come from the Bridge specification's byte-placement rules
//! (`docs/Octane/`) and another Octane emulator's register tests, not from
//! reading the Rust back. Where a layout is derived rather than copied, the
//! rule is written next to it.
//!
//! RAM placement rules used below, for the 32-bit slots the Octane has:
//!
//! * Bridge, DMA swap clear: PCI byte at address a is RAM byte a ^ 3. Swap
//!   set: RAM byte a. A card that writes little-endian words (the ISP1040)
//!   therefore leaves big-endian words in RAM with the swap clear.
//! * IOC3 hands the Bridge its internal byte a as PCI byte a ^ 3, so with the
//!   swap clear its structures and frames sit in RAM in their own order, and
//!   with the swap set every aligned word is reversed.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use super::*;
use crate::dev::bridge::*;
use crate::dev::heart::{Heart, HeartConfig, PIU_BASE, PIU_SIZE};
use crate::physical::{read_bytes, write_bytes, Physical};
use crate::dev::ioc3::{self, Ioc3};
use crate::dev::ioc3_eth::{bits as ebit, reg as ereg, Ioc3Eth};
use crate::dev::isp1040::{mbox, reg as ireg, Isp1040, CTRL_INT_ENABLE, CTRL_RISC_INT_ENABLE, ENTRY_SIZE, ENTRY_TYPE_REQUEST, ENTRY_TYPE_STATUS};
use crate::dev::mgras::hq4::ImpactXio;
use crate::dev::xbow::Xbow;
use crate::traits::{BusDevice, Device, Resettable, BUS_ERR, BUS_OK};

const FLASH_SIZE: usize = 1 << 20;
const XBOW_REVISION: u32 = 4;
const WIDGET_GFX: u8 = 11;

// ---- 0. the processor's bus ----

/// The bus as the machine builds it, with no graphics: 128 MiB in bank 0,
/// HEART, the Crossbow (HEART and the Bridge alive), the Bridge with `prom`
/// in its flash.
struct Board {
    bus: Arc<Ip30Bus>,
    phys: Arc<Physical>,
    heart: Arc<Heart>,
}

fn board(prom: &[u8]) -> Board {
    board_with(prom, [128, 0, 0, 0])
}

/// `board` with the DIMMs fitted in each bank, in MB.
fn board_with(prom: &[u8], banks_mb: [u32; 4]) -> Board {
    let ram = Arc::new(HeartRam::new(banks_mb));
    let heart = Arc::new(Heart::new(HeartConfig::octane(banks_mb.map(|mb| (mb as u64) << 20)), crate::dev::heart::host_clock()));
    let xbow = Arc::new(Xbow::new(XBOW_REVISION, 1 << WIDGET_HEART | 1 << WIDGET_BRIDGE));
    xbow.connect_heart(heart.clone());
    let bridge = Bridge::new(ram.dma_view(), heart.clone());
    bridge.attach_flash(Arc::new(RawFlash::new(prom, FLASH_SIZE)), FLASH_SIZE as u32);
    let phys = Ip30Bus::new(ram);
    phys.map(PIU_BASE, PIU_BASE + PIU_SIZE, heart.clone(), "heart piu");
    phys.map(widget_small_base(WIDGET_XBOW), widget_small_base(WIDGET_XBOW) + WIDGET_SMALL_SIZE, xbow.clone(), "xbow");
    phys.map(widget_small_base(WIDGET_HEART), widget_small_base(WIDGET_HEART) + WIDGET_SMALL_SIZE, heart.clone(), "heart widget");
    phys.attach_widget(WIDGET_BRIDGE, bridge, false);
    let phys = Arc::new(phys);
    phys.init();
    for w in 9..16 {
        phys.set_widget_link(w, xbow.carries(w));
    }
    let weak = Arc::downgrade(&phys);
    heart.on_memcfg(Box::new(move |m| {
        if let Some(p) = weak.upgrade() {
            p.remap_banks(m);
        }
    }));
    Board { phys: phys.phys.clone(), bus: phys, heart }
}

/// The PROM stores to its own variables at their cached flash addresses.
/// Through the CPU's cache the store completes and a later cached load
/// sees it; the flash, read uncached, still holds the image, and a reset
/// forgets the line.
#[test]
fn cached_stores_to_the_flash_stay_in_the_cache() {
    use crate::cpu::mips_cache_shadow::R10000ShadowCache;
    use crate::cpu::mips_cache_v2::MipsCache;
    let mut prom = vec![0u8; 0x2000];
    prom[0x1140..0x1150].copy_from_slice(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    let b = board(&prom);
    let flash: Arc<dyn BusDevice> = b.phys.clone();
    let lines = Arc::new(CachedFlashLines::new(flash.clone()));
    let cache = R10000ShadowCache::from(lines.clone() as Arc<dyn BusDevice>);
    let (va, pa) = (0xffff_ffff_9fc0_1148, 0x1fc0_1148);

    assert_eq!(flash.write32(pa, 1), BUS_ERR, "the Bridge's flash port refuses a 4-byte write");
    assert_eq!(cache.write::<4>(va, pa, 1), BUS_OK, "a cached store completes");
    assert_eq!(cache.read::<4>(va, pa).data, 1, "a later cached load sees it");
    assert_eq!(cache.read::<4>(va - 8, pa - 8).data, 0xffff_ffff, "the rest of the line is the flash's");
    assert_eq!(cache.read::<8>(va, pa).data, 0x0000_0001_0000_0001, "wider loads see the line too");
    assert_eq!(flash.read32(pa).data, 0, "the flash, read uncached, is untouched");
    assert_eq!(cache.read::<4>(va + 0x80, pa + 0x80).data, 0, "the next line is not held");
    let mut line = [0u64; 2];
    assert_eq!(lines.read_block(pa & !0xf, &mut line), BUS_OK);
    assert_eq!(line[1], 0x0000_0001_0000_0001, "a line fill sees the held line");

    lines.clear();
    assert_eq!(cache.read::<4>(va, pa).data, 0, "reset forgets the line");
}

#[test]
fn the_reset_vector_reads_the_prom() {
    let b = board(&[0x0b, 0xf0, 0x00, 0xf0, 0, 0, 0, 0]);
    let p = &b.phys;
    assert_eq!(p.read32(0x1fc0_0000).data, 0x0bf0_00f0);
    assert_eq!(p.read64(0x1fc0_0000).data, 0x0bf0_00f0_0000_0000);
    assert_eq!(p.read8(0x1fc0_0001).data, 0xf0);
    assert_eq!(p.read32(0x1fc0_0008).data, 0xffff_ffff, "erased past the image");
    assert_eq!(p.read32(0x1f00_0004).data >> 12 & 0xffff, 0xc002, "the Bridge's widget identity");
    // The CPU's attribute bits above 40 fold away.
    assert_eq!(p.read32(0xff00_0000_1fc0_0000).data, 0x0bf0_00f0);
}

#[test]
fn low_memory_is_the_first_16_kib_of_sdram_on_the_bus() {
    let b = board(&[]);
    let p = &b.phys;
    // Bank 0, 128 MiB at the start of SDRAM space.
    assert_eq!(p.write32(PIU_BASE + 0x20, 0x8003_0000), BUS_OK);
    // The processor reaches those bytes only at physical 0; a bus master's
    // view shows where they are.
    let dma = b.bus.memory();
    assert_eq!(p.read64(0).data, 0);
    assert_eq!(p.write32(0x100, 0x1234_5678), BUS_OK);
    assert_eq!(dma.read32(0x2000_0100).data, 0x1234_5678, "a low store is in SDRAM");
    assert_eq!(p.read32(0x2000_0100).status, BUS_ERR, "and the processor cannot see it there");
    assert_eq!(p.read16(0x102).data, 0x5678);
    assert_eq!(p.read8(0x100).data, 0x12);
    assert_eq!(dma.write64(0x2000_0200, 0x0102_0304_0506_0708), BUS_OK);
    assert_eq!(p.read64(0x200).data, 0x0102_0304_0506_0708, "and the reverse");
    assert_eq!(p.mem_ptr(0x100), dma.mem_ptr(0x2000_0100), "one word");
    // The ppmem window maps whole 32 MiB units, so `mem_ptr` (the cache's
    // line-fill path) reaches the guard's bytes, as the cache always has;
    // the processor's uncached accesses there are bus errors.
    assert_eq!(p.write64(0x2000_0000, 1), BUS_ERR);
    assert_eq!(p.read8(0x2000_3fff).status, BUS_ERR);
    assert_eq!(p.read8(0x3fff).status, BUS_OK);
    assert_eq!(p.read8(0x4000).status, BUS_ERR);
    assert_eq!(p.read8(0x10000).status, BUS_ERR);
    assert_eq!(p.write64(0x2000_4000, 1), BUS_OK);
    assert_eq!(p.read64(0x4000).status, BUS_ERR, "SDRAM past 16 KiB is not aliased");
    assert!(b.bus.unmapped().iter().any(|&(a, w, _)| a == 0x4000 && !w), "misses are logged");
}

/// Code in low memory and SDRAM is compiled against a per-page
/// generation that every store into the page advances.
#[cfg(feature = "jitv2")]
#[test]
fn stores_through_the_bus_invalidate_compiled_code() {
    let b = board(&[]);
    let p = &b.phys;
    let gen = |g: *const AtomicU64| unsafe { (*g).load(Ordering::Acquire) };
    assert_eq!(p.write32(PIU_BASE + 0x20, 0x8003_0000), BUS_OK);
    let (g0, g1) = (p.gen_ptr(0x180), p.gen_ptr(0x1000));
    assert!(!g0.is_null() && !g1.is_null() && g0 != g1);
    let dma = b.bus.memory();
    assert_eq!(g0, dma.gen_ptr(0x2000_0180), "low memory shares SDRAM's counter");
    // At the SDRAM address the window's counter answers (the window maps
    // whole 32 MiB units); it is the same memory, so it moves with g0.
    let (a, c) = (gen(g0), gen(g1));
    assert_eq!(p.write64(0x180, 0xdf5a_c000_0340_0008), BUS_OK);
    assert_ne!(gen(g0), a, "a store advances its page");
    assert_eq!(gen(g1), c, "and no other");
    let v = gen(g0);
    assert_eq!(dma.write8(0x2000_0181, 1), BUS_OK);
    assert_ne!(gen(g0), v, "a bus master's store at the SDRAM address advances it too");
    assert!(p.gen_ptr(0x4000).is_null(), "the hole after low memory has none");
    let g = p.gen_ptr(0x2000_8000);
    let v = gen(g);
    assert_eq!(p.write8(0x2000_8001, 1), BUS_OK);
    assert_ne!(gen(g), v);
    assert!(p.gen_ptr(0x1fc0_0000).is_null(), "the flash has none");
}

#[test]
fn sdram_follows_the_descriptors() {
    let b = board(&[]);
    let p = &b.phys;
    assert_eq!(p.read64(0x2000_4000).status, BUS_ERR, "no descriptor valid at reset");
    // Bank 0, 128 MiB at the start of SDRAM space.
    assert_eq!(p.write32(PIU_BASE + 0x20, 0x8003_0000), BUS_OK);
    assert_eq!(p.read64(0x2000_0000).status, BUS_ERR, "SDRAM's first 16 KiB are low memory's");
    assert_eq!(p.read64(0x2000_3ff8).status, BUS_ERR);
    assert_eq!(p.read64(0x2000_4000).data, 0, "SDRAM from 16 KiB up");
    assert_eq!(p.write64(0x2000_4000, 0x0102_0304_0506_0708), BUS_OK);
    assert_eq!(p.read32(0x2000_4004).data, 0x0506_0708);
    assert_eq!(p.read64(0x2800_0000).status, BUS_ERR, "past the bank");
    let ptr = p.mem_ptr(0x2000_4000).expect("memory has a direct pointer");
    assert_eq!(unsafe { *ptr }.rotate_left(32), 0x0102_0304_0506_0708);
    // The PROM's probe: bank 0 as 2 GiB at 2 GiB up, above 4 GiB in part.
    assert_eq!(p.write32(PIU_BASE + 0x20, 0x81bf_0040), BUS_OK);
    assert_eq!(p.read64(0x2000_4000).status, BUS_ERR, "the old place is gone");
    assert_eq!(p.write64(0x1_1000_0000, 7), BUS_OK, "above 4 GiB through the bus");
    assert_eq!(p.read64(0x1_1000_0000).data, 7);
    // HEART's reset takes the descriptors, and memory, away.
    crate::traits::Resettable::power_on(&*b.heart);
    assert_eq!(p.read64(0x1_1000_0000).status, BUS_ERR);
}

// The memory map as the processor and a bus master see it. Each of these
// holds on the bus path; the one that does not hold for a cached access is
// named as a limitation.

/// Write HEART's descriptor for `bank`, as the PROM does through the PIU.
fn set_bank(p: &Physical, bank: u32, desc: u32) {
    assert_eq!(p.write32(PIU_BASE + 0x20 + 4 * bank as u64, desc), BUS_OK);
}

#[test]
fn map_a_valid_descriptor_over_an_empty_bank_reads_zero_and_drops_writes() {
    let b = board(&[]);
    let p = &b.phys;
    let dma = b.bus.memory();
    // Bank 1 is not fitted. Its descriptor claims 64 MiB at the third unit.
    set_bank(p, 1, 0x8001_0002);
    let at = 0x2000_0000 + 2 * 0x200_0000 + 0x100;
    assert_eq!(p.write64(at, u64::MAX), BUS_OK, "the write is accepted");
    assert_eq!(p.read64(at).data, 0, "and goes nowhere");
    assert_eq!(p.read8(at).status, BUS_OK);
    assert_eq!(dma.read64(at).status, BUS_ERR, "a bus master gets an error");
    assert_eq!(p.read64(at + 0x400_0000).status, BUS_ERR, "past the descriptor's span");
    // With no other descriptor over it, the start of SDRAM is empty memory
    // too: low memory answers as that bank does, past the guard.
    set_bank(p, 1, 0x8001_0000);
    assert_eq!(p.write32(0x100, 5), BUS_OK);
    assert_eq!(p.read32(0x100).data, 0);
    assert_eq!(p.read8(0x2000_0000).status, BUS_ERR, "the guard still faults");
    assert_eq!(p.read32(0x2000_4000).data, 0);
    assert_eq!(dma.read32(0x2000_0100).status, BUS_ERR);
}

#[test]
fn map_the_proms_bank_size_probe_gets_each_size_s_answer() {
    // The stock PROM sizes a bank through a temporary 2 GiB descriptor
    // (0x81bf0040), writing patterns and watching which address bits alias,
    // then looks the result up. These are the descriptors it settles on.
    const PROBE: u64 = 0x2000_0000 + 0x40 * 0x200_0000;
    const UNIT: u64 = 0x200_0000;
    for (mib, want) in [(64u32, 0x8101_0040u32), (128, 0x8103_0040), (256, 0x8147_0040)] {
        let b = board_with(&[], [mib, 0, 0, 0]);
        let p = &b.phys;
        set_bank(p, 0, 0x81bf_0040);
        let distinct = |step: u64, pat: u64| {
            assert_eq!(p.write64(PROBE, pat), BUS_OK);
            assert_eq!(p.write64(PROBE + step, !pat), BUS_OK);
            p.read64(PROBE).data == pat
        };
        let mut bits = 0;
        assert_eq!(p.write64(PROBE + 0x100, 0x1111_1111_1111_1111), BUS_OK);
        assert_eq!(p.write64(PROBE, !0x1111_1111_1111_1111), BUS_OK);
        if p.read64(PROBE + 0x100).data == 0x1111_1111_1111_1111 {
            bits |= 0x20;
        }
        if distinct(4 * UNIT, 0x2222_2222_2222_2222) {
            bits |= 0x10;
        } else if distinct(UNIT, 0x4444_4444_4444_4444) {
            bits |= 0x08;
        }
        if distinct(32 * UNIT, 0x8888_8888_8888_8888) {
            bits |= 0x04;
        }
        // The PROM's table, indexed by the three bits it found.
        const TABLE: [u32; 16] = [
            0x0000_0040, 0x0001_0040, 0x0043_0040, 0x0047_0040, 0x008f_0040, 0x009f_0040, 0x0000_0040, 0x0000_0040,
            0x0101_0040, 0x0103_0040, 0x0147_0040, 0x014f_0040, 0x019f_0040, 0x01bf_0040, 0x0000_0040, 0x0000_0040,
        ];
        let word = TABLE[bits / 4] | 0x8000_0000;
        assert_eq!(word, want, "{mib} MiB");
        set_bank(p, 0, word);
        assert_eq!(b.heart.memcfg()[0], want);
        // The descriptor it settles on holds the fitted memory where the probe was, nothing past it.
        let end = PROBE + ((mib as u64) << 20);
        assert_eq!(p.write64(end - 8, 3), BUS_OK, "{mib} MiB: last doubleword");
        assert_eq!(p.read64(end - 8).data, 3);
        assert_eq!(p.read64(end).status, BUS_ERR, "{mib} MiB: past the end");
    }
}

#[test]
fn map_a_bank_above_4_gib_is_reached_through_the_bus_only() {
    let b = board_with(&[], [64, 64, 0, 0]);
    let p = &b.phys;
    let dma = b.bus.memory();
    // Bank 0 at the start of SDRAM space; bank 1 at unit 0x80, 4 GiB up.
    set_bank(p, 0, 0x8001_0000);
    set_bank(p, 1, 0x8001_0080);
    let high = 0x2000_0000 + 0x80 * 0x200_0000;
    assert_eq!(high, 0x1_2000_0000);
    assert_eq!(p.write64(high + 0x8000, 0x0123_4567_89ab_cdef), BUS_OK);
    assert_eq!(p.read64(high + 0x8000).data, 0x0123_4567_89ab_cdef);
    assert_eq!(dma.read64(high + 0x8000).data, 0x0123_4567_89ab_cdef);
    assert_eq!(p.read64(0x2000_8000).data, 0, "bank 0 is a different bank");
    assert!(p.mem_ptr(high + 0x8000).is_some(), "it has a direct pointer");
    assert_eq!(p.read64(high + 0x400_0000).status, BUS_ERR, "past the bank");
}

#[test]
fn map_a_bus_master_reaches_the_guarded_bytes() {
    let b = board(&[]);
    let p = &b.phys;
    set_bank(p, 0, 0x8003_0000);
    let dma = b.bus.memory();
    assert_eq!(p.read64(0x2000_0000).status, BUS_ERR);
    assert_eq!(p.write64(0x2000_3000, 1), BUS_ERR);
    assert_eq!(dma.write64(0x2000_3000, 0x0a0b_0c0d_0e0f_1011), BUS_OK);
    assert_eq!(dma.read64(0x2000_3000).data, 0x0a0b_0c0d_0e0f_1011);
    assert_eq!(p.read64(0x3000).data, 0x0a0b_0c0d_0e0f_1011, "the same bytes, at low memory");
    let mut bytes = [0u8; 8];
    assert_eq!(read_bytes(&*dma, 0x2000_3000, &mut bytes), BUS_OK);
    assert_eq!(bytes, [0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11]);
}

/// Low memory and SDRAM are one page-generation counter, whichever way a
/// store comes in: through the alias, at the SDRAM address by a bus master,
/// or through a block write.
#[cfg(feature = "jitv2")]
#[test]
fn map_the_alias_and_sdram_share_page_generations() {
    let b = board(&[]);
    let p = &b.phys;
    set_bank(p, 0, 0x8003_0000);
    let dma = b.bus.memory();
    let gen = |g: *const AtomicU64| unsafe { (*g).load(Ordering::Acquire) };
    let low = p.gen_ptr(0x2000);
    assert!(!low.is_null());
    assert_eq!(low, dma.gen_ptr(0x2000_2000));
    let v = gen(low);
    assert_eq!(p.write_block(0x2000, &[1, 2]), BUS_OK);
    assert_ne!(gen(low), v, "a block store through the alias");
    let v = gen(low);
    assert_eq!(dma.write_block(0x2000_2000, &[3]), BUS_OK);
    assert_ne!(gen(low), v, "a bus master's block store at the SDRAM address");
}

/// Limitation: real HEART faults cached accesses to the guard as well, but
/// the CPU's fast paths go through the ppmem window, which maps the whole
/// 32 MiB unit, guard included.
#[test]
fn limitation_the_window_reaches_the_guard() {
    let b = board(&[]);
    set_bank(&b.phys, 0, 0x8003_0000);
    assert_eq!(b.phys.read32(0x2000_3000).status, BUS_ERR, "uncached reads fault");
    assert!(b.phys.mem_ptr(0x2000_3000).is_some(), "limitation: the window path reaches the bytes");
}

#[test]
fn the_miss_list_keeps_the_first_64_and_then_stops() {
    let b = board(&[]);
    let p = &b.phys;
    for i in 0..100u64 {
        assert_eq!(p.read32(0xf0_0000_0000 + i * 0x1_0000).status, BUS_ERR);
    }
    let u = b.bus.unmapped();
    assert_eq!(u.len(), 64);
    assert_eq!(u[63].0, 0xf0_0000_0000 + 63 * 0x1_0000, "the first 64, in order");
    assert_eq!(p.write32(0xf1_0000_0000, 0), BUS_ERR);
    assert_eq!(b.bus.unmapped(), u, "a full list is left alone");
}

#[test]
fn widgets_route_by_number() {
    let b = board(&[]);
    let p = &b.phys;
    assert_eq!(p.read32(0x1000_0004).data, 0x4000_0001, "Crossbow");
    assert_eq!(p.read32(0x1800_0004).data, 0x4c00_106d, "HEART");
    assert_eq!(p.read64(0x1800_0000).data, 0x4c00_106d);
    assert_eq!(p.read32(0x1800_1004).status, BUS_ERR, "no route back to HEART");
    assert_eq!(p.read32(0x1b00_0004).status, BUS_ERR, "an empty graphics port");
    assert_eq!(p.read32(0x1100_0004).status, BUS_ERR, "nothing on widget 1");
    assert_eq!(p.read64(0x0ff1_0030).data, 0, "HEART's pending bits");
    assert_eq!(p.read32(0x0ff0_0020).data, 0);
    assert_eq!(p.read32(widget_large_base(WIDGET_BRIDGE)).status, BUS_ERR, "the Bridge takes no large window");
}

#[test]
fn the_graphics_port_reaches_impact_with_its_link_up() {
    let ram = Arc::new(HeartRam::new([128, 0, 0, 0]));
    let heart = Arc::new(Heart::new(HeartConfig::octane([128 << 20, 0, 0, 0]), crate::dev::heart::host_clock()));
    let gfx = ImpactXio::new(
        WIDGET_GFX,
        heart.clone(),
        ram.dma_view(),
        crate::disp::StatusSource::detached(),
    );
    let id = widget_small_base(WIDGET_GFX) + 4;
    for (links, alive) in [(0x8100u16, false), (0x8100 | 1 << WIDGET_GFX, true)] {
        let xbow = Arc::new(Xbow::new(XBOW_REVISION, links));
        xbow.connect_heart(heart.clone());
        let p = Arc::new(Ip30Bus::new(ram.clone()));
        p.attach_widget(WIDGET_GFX, gfx.clone(), false);
        p.init();
        let weak = Arc::downgrade(&p);
        xbow.on_link_change(Box::new(move |w, up| {
            if let Some(p) = weak.upgrade() {
                p.set_widget_link(w, up);
            }
        }));
        for w in 9..16 {
            p.set_widget_link(w, xbow.carries(w));
        }
        let r = p.phys.read32(id);
        if alive {
            assert_eq!(r.status, BUS_OK);
            assert_eq!(r.data >> 12 & 0xffff, 0xc003, "HQ4's widget part number");
        } else {
            assert_eq!(r.status, BUS_ERR, "no link, no route");
        }
        // The Crossbow attaching a board brings its windows up.
        xbow.attach(WIDGET_GFX);
        assert_eq!(p.phys.read32(id).status, BUS_OK, "a link that comes up is routed");
    }
}

// ---- the devices behind the Bridge ----

const RAM: u64 = 0x2000_0000;
/// Where a card sees RAM through the direct map once the map targets HEART.
const PCI_RAM: u32 = 0xa000_0000;

const SLOT_ISP0: u32 = 0;
const SLOT_ISP1: u32 = 1;
const SLOT_IOC3: u32 = 2;

const MAC: [u8; 6] = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];

/// Bridge input the SuperIO block of IOC3 is wired to, and the Ethernet's
/// (the PCI INTA of slot 2).
const SUPERIO_PIN: u32 = 4;
const ENET_PIN: u32 = 2;

struct Octane {
    heart: Arc<Heart>,
    mem: Arc<dyn BusDevice>,
    bridge: Arc<Bridge>,
    isp: [Arc<Isp1040>; 2],
    sent: rtrb::Consumer<Vec<u8>>,
    inject: rtrb::Producer<Vec<u8>>,
    eth: Arc<Ioc3Eth>,
}

impl Octane {
    fn new() -> Self {
        let ram = Arc::new(HeartRam::new([64, 0, 0, 0]));
        // One valid descriptor: bank 0, two 32 MB units from 0x2000_0000.
        ram.set_memcfg([0x8001_0000, 0, 0, 0]);
        let mem = ram.dma_view();
        let heart = Arc::new(Heart::new(HeartConfig::octane([64 << 20, 0, 0, 0]), Arc::new(|| 0)));
        let bridge = Bridge::new(mem.clone(), heart.clone());

        let eth = Ioc3Eth::with_negotiation_delay(MAC, std::time::Duration::ZERO);
        let (link, sent, inject) = crate::net::Link::loopback();
        eth.attach(link);
        let ioc3 = Arc::new(Ioc3::octane(MAC, false));
        ioc3.attach_eth(eth.clone());
        ioc3.connect_superio_interrupt(bridge.pin_line(SUPERIO_PIN as u8));

        let isp = [Arc::new(Isp1040::new_at(0, Isp1040::new_bus())), Arc::new(Isp1040::new_at(1, Isp1040::new_bus()))];
        bridge.attach(SLOT_ISP0 as u8, isp[0].clone());
        bridge.attach(SLOT_ISP1 as u8, isp[1].clone());
        bridge.attach(SLOT_IOC3 as u8, ioc3.clone());
        Octane { heart, mem, bridge, isp, sent, inject, eth }
    }

    // ---- the processor's side ----

    fn set(&self, off: u32, val: u32) {
        assert_eq!(self.bridge.write32(u64::from(off + 4), val), BUS_OK);
    }

    fn get(&self, off: u32) -> u32 {
        let r = self.bridge.read32(u64::from(off + 4));
        assert!(r.is_ok());
        r.data
    }

    fn cfg_w32(&self, slot: u32, reg: u32, val: u32) {
        assert_eq!(self.bridge.write32(u64::from(CFG0_BASE + slot * 0x1000 + reg), val), BUS_OK);
    }

    fn cfg_r32(&self, slot: u32, reg: u32) -> u32 {
        let r = self.bridge.read32(u64::from(CFG0_BASE + slot * 0x1000 + reg));
        assert!(r.is_ok());
        r.data
    }

    /// Size and place BAR `bar` of `slot` at PCI `base`, enable it (and bus
    /// mastering), and aim the slot's device window at memory space with
    /// `dev_bits` added. This is the PROM's sequence in the qtests.
    fn map(&self, slot: u32, bar: u32, base: u32, dev_bits: u32) {
        self.cfg_w32(slot, 0x10 + 4 * bar, base);
        self.cfg_w32(slot, 0x04, 0x6);
        self.set(reg::device(slot), 0x1000 | (base >> 20) | dev_bits);
    }

    fn direct_map_to_heart(&self) {
        self.set(reg::DIRECT_MAP, (WIDGET_HEART as u32) << DIRECT_WIDGET_SHIFT);
    }

    /// Send `pin`'s interrupts to HEART as `vector`, with a clear packet when
    /// the pin falls, and enable the pin.
    fn route(&self, pin: u32, vector: u32) {
        self.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
        self.set(reg::INT_DEST_LOWER, 0x80);
        self.set(reg::int_host(pin), vector);
        let mode = self.get(reg::INT_MODE);
        self.set(reg::INT_MODE, mode | 1 << pin);
        let en = self.get(reg::INT_ENABLE);
        self.set(reg::INT_ENABLE, en | 1 << pin);
    }

    fn pending(&self) -> u64 {
        self.heart.pending()
    }

    // ---- IOC3 at its slot's window ----

    fn ioc3_win() -> u32 {
        device_window_offset(SLOT_IOC3)
    }

    fn ioc3_map(&self) {
        self.map(SLOT_IOC3, 0, 0x1000_0000, 0);
    }

    fn i8(&self, off: u32) -> u8 {
        let r = self.bridge.read8(u64::from(Self::ioc3_win() + off));
        assert!(r.is_ok(), "read8 at IOC3 +{off:#x}");
        r.data
    }

    fn i16(&self, off: u32) -> u16 {
        let r = self.bridge.read16(u64::from(Self::ioc3_win() + off));
        assert!(r.is_ok(), "read16 at IOC3 +{off:#x}");
        r.data
    }

    fn i32(&self, off: u32) -> u32 {
        let r = self.bridge.read32(u64::from(Self::ioc3_win() + off));
        assert!(r.is_ok(), "read32 at IOC3 +{off:#x}");
        r.data
    }

    fn w8(&self, off: u32, v: u8) {
        assert_eq!(self.bridge.write8(u64::from(Self::ioc3_win() + off), v), BUS_OK);
    }

    fn w16(&self, off: u32, v: u16) {
        assert_eq!(self.bridge.write16(u64::from(Self::ioc3_win() + off), v), BUS_OK);
    }

    fn w32(&self, off: u32, v: u32) {
        assert_eq!(self.bridge.write32(u64::from(Self::ioc3_win() + off), v), BUS_OK);
    }

    // ---- RAM, as the bytes sit in memory ----

    fn ram_put(&self, off: u64, data: &[u8]) {
        assert_eq!(write_bytes(&*self.mem, RAM + off, data), BUS_OK);
    }

    fn ram_get(&self, off: u64, n: usize) -> Vec<u8> {
        let mut v = vec![0; n];
        assert_eq!(read_bytes(&*self.mem, RAM + off, &mut v), BUS_OK);
        v
    }

    /// Bytes `data` as a card with PCI address `at` sees them, written into
    /// RAM the way the CPU left them for a card on a 32-bit slot. `isp_like`
    /// cards (plain little-endian bus masters) see RAM byte a ^ 3 with the
    /// swap clear; IOC3 sees RAM byte a with the swap clear.
    fn place(&self, off: u64, data: &[u8], reversed: bool) {
        for (i, &b) in data.iter().enumerate() {
            let a = off + i as u64;
            self.ram_put(if reversed { a ^ 3 } else { a }, &[b]);
        }
    }

    fn unplace(&self, off: u64, n: usize, reversed: bool) -> Vec<u8> {
        (0..n as u64).map(|i| self.ram_get(if reversed { (off + i) ^ 3 } else { off + i }, 1)[0]).collect()
    }
}

// ---- 1. IOC3 serial and the DS1685 through the Bridge window ----

const UART_A: u32 = ioc3::reg::SERIAL_A;
const RTC_INDEX: u32 = ioc3::reg::BYTEBUS_DEV1;
const RTC_DATA: u32 = ioc3::reg::BYTEBUS_DEV2;

#[test]
fn ioc3_serial_loopback_through_the_window_by_byte() {
    // octane-test.c, test_serial_a_loopback and test_pci_bar_separation.
    let o = Octane::new();
    o.ioc3_map();
    assert_eq!(o.cfg_r32(SLOT_IOC3, 0), 0x0003_10a9, "IOC3's vendor and device");
    o.w8(UART_A + 4, 0x10);
    o.w8(UART_A, 0x5a);
    assert_eq!(o.i8(UART_A), 0x5a);
    o.w8(UART_A + 7, 0x5a);
    assert_eq!(o.i8(UART_A + 7), 0x5a);
    o.w8(UART_A + 3, 0x03);
    assert_eq!(o.i8(UART_A + 3), 0x03);
    // Each byte lane of a register word reaches a different register.
    assert_eq!(o.i8(UART_A + 4), 0x10);
}

#[test]
fn ioc3_serial_halfword_and_word_accesses_are_little_endian_runs_of_registers() {
    // The UART is byte-wide. A wider processor access is several registers,
    // lowest register address in the least significant byte, because the
    // Bridge's swap-clear placement makes every access little-endian on PCI.
    let o = Octane::new();
    o.ioc3_map();
    o.w8(UART_A + 4, 0x10);
    o.w8(UART_A + 3, 0x03);
    o.w8(UART_A + 7, 0x5a);
    let _ = o.i8(UART_A + 6); // clear the modem status change bits
    let (iir, lcr) = (o.i8(UART_A + 2), o.i8(UART_A + 3));
    assert_eq!(lcr, 0x03);
    assert_eq!(o.i16(UART_A + 2), u16::from(lcr) << 8 | u16::from(iir));
    let (mcr, lsr, msr, scr) = (o.i8(UART_A + 4), o.i8(UART_A + 5), o.i8(UART_A + 6), o.i8(UART_A + 7));
    assert_eq!((mcr, scr), (0x10, 0x5a));
    assert_eq!(o.i32(UART_A + 4), u32::from_le_bytes([mcr, lsr, msr, scr]));

    // Writes: the low byte goes to the lower register. MSR and LSR are
    // read-only, so only MCR and SCR change.
    o.w32(UART_A + 4, 0x3c00_0000 | 0x10);
    assert_eq!((o.i8(UART_A + 4), o.i8(UART_A + 7)), (0x10, 0x3c));
    o.w16(UART_A + 6, 0xa600);
    assert_eq!(o.i8(UART_A + 7), 0xa6);
    assert_eq!(o.i8(UART_A + 4), 0x10);
}

#[test]
fn ioc3_ds1685_ports_answer_every_width_and_lane() {
    // octane-test.c, test_storage_nvram: register number to the first port,
    // the register on the second. Each port is a byte-wide select that
    // answers on any lane of its first word (ds1685.c ignores the address).
    let o = Octane::new();
    o.ioc3_map();
    o.w8(RTC_INDEX, 0x0e);
    o.w8(RTC_DATA, 0x5a);
    for lane in 0..4 {
        assert_eq!(o.i8(RTC_DATA + lane), 0x5a, "lane {lane}");
    }
    assert_eq!(o.i8(RTC_INDEX), 0x0e, "the model answers a read of the address port with its latch");
    // A halfword or word read is the same register once per lane.
    assert_eq!(o.i16(RTC_DATA), 0x5a5a);
    assert_eq!(o.i32(RTC_DATA), 0x5a5a_5a5a);
    // Another lane writes the same register.
    o.w8(RTC_DATA + 3, 0x77);
    assert_eq!(o.i8(RTC_DATA), 0x77);
    // A halfword store is two byte cycles, low byte first: the last wins.
    o.w16(RTC_DATA, 0x1234);
    assert_eq!(o.i8(RTC_DATA), 0x12);
    o.w32(RTC_DATA, 0xaabb_ccdd);
    assert_eq!(o.i8(RTC_DATA), 0xaa);
    // The address latch keeps seven bits, and a word store lands the same way.
    o.w32(RTC_INDEX, 0x0e0e_0e8f);
    assert_eq!(o.i8(RTC_INDEX), 0x0e);
    // Nothing else in the chip moved: the other UART is untouched.
    assert_eq!(o.i8(ioc3::reg::SERIAL_B + 7), 0);
}

// ---- 2. ISP1040 ----

struct IspDisk(std::path::PathBuf);

impl IspDisk {
    fn new(name: &str, isp: &Isp1040, target: u8) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let p = std::env::temp_dir().join(format!("iris-ip30-xtests-{}-{}-{name}.img", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::write(&p, vec![0u8; 64 * 512]).unwrap();
        isp.bus().attach_image(target, &p, false).unwrap();
        IspDisk(p)
    }
}

impl Drop for IspDisk {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl Octane {
    fn isp_win(slot: u32) -> u32 {
        device_window_offset(slot)
    }

    /// A 16-bit register access with the PIO swap clear: a halfword to a
    /// 32-bit device goes to offset ^ 2 (octane-test.c ISP_MAILBOX).
    fn isp_w(&self, slot: u32, r: u32, v: u16) {
        assert_eq!(self.bridge.write16(u64::from(Self::isp_win(slot) + (r ^ 2)), v), BUS_OK);
    }

    fn isp_r(&self, slot: u32, r: u32) -> u16 {
        let x = self.bridge.read16(u64::from(Self::isp_win(slot) + (r ^ 2)));
        assert!(x.is_ok(), "read16 at ISP {r:#x}");
        x.data
    }

    fn isp_cmd(&self, slot: u32, regs: &[u16]) -> u16 {
        for (i, &v) in regs.iter().enumerate() {
            self.isp_w(slot, ireg::MAILBOX_BASE + 2 * i as u32, v);
        }
        self.isp_w(slot, ireg::HOST_CMD, 0x5000);
        let status = self.isp_r(slot, ireg::MAILBOX_BASE);
        self.isp_w(slot, ireg::HOST_CMD, 0x7000);
        self.isp_w(slot, ireg::SEMAPHORE, 0);
        status
    }

    fn isp_queue(&self, slot: u32, cmd: u16, len: u16, pci: u32) -> u16 {
        self.isp_cmd(slot, &[cmd, len, (pci >> 16) as u16, pci as u16])
    }
}

#[test]
fn isp_sixteen_bit_registers_through_the_window() {
    let o = Octane::new();
    o.map(SLOT_ISP0, 1, 0x1000_0000, 0);
    // Identity, read as halfwords at the lane the Bridge places them on.
    assert_eq!(o.isp_r(SLOT_ISP0, ireg::ID_LOW), 0x1077);
    assert_eq!(o.isp_r(SLOT_ISP0, ireg::ID_HIGH), 0x1020);
    // The same halfwords at the unadjusted address reach the other half of
    // the word, which is how a driver that forgot the lane swap would see it.
    let w = Octane::isp_win(SLOT_ISP0);
    assert_eq!(o.bridge.read16(u64::from(w)).data, 0x1020);
    assert_eq!(o.bridge.read16(u64::from(w + 2)).data, 0x1077);
    // Mailboxes round trip.
    for n in 0..8u32 {
        o.isp_w(SLOT_ISP0, ireg::MAILBOX_BASE + 2 * n, 0x1000 + n as u16);
    }
    // (Mailboxes 4 and 5 are the ring pointers, so only 0..4 read back as
    // plain storage here.)
    for n in 0..4u32 {
        assert_eq!(o.isp_r(SLOT_ISP0, ireg::MAILBOX_BASE + 2 * n), 0x1000 + n as u16);
    }
    // The second controller is a separate window.
    o.map(SLOT_ISP1, 1, 0x1100_0000, 0);
    o.isp_w(SLOT_ISP1, ireg::MAILBOX_BASE, 0x1111);
    assert_eq!(o.isp_r(SLOT_ISP1, ireg::MAILBOX_BASE), 0x1111);
    assert_eq!(o.isp_r(SLOT_ISP0, ireg::MAILBOX_BASE), 0x1000);
    // Only sixteen-bit cycles are decoded: a byte read does not reach a register.
    assert_eq!(o.bridge.read8(u64::from(w + 3)).data, 0);
}

#[test]
fn isp_programmed_pio_swap_makes_the_window_a_byte_stream() {
    // Bridge PIO swap set: host byte k is PCI byte k, no lane adjustment, so
    // a big-endian halfword of a little-endian register is byte-swapped.
    let o = Octane::new();
    o.map(SLOT_ISP0, 1, 0x1000_0000, DEV_SWAP_PROGRAMMED);
    let w = Octane::isp_win(SLOT_ISP0);
    assert_eq!(o.bridge.read16(u64::from(w)).data, 0x7710);
    assert_eq!(o.bridge.read16(u64::from(w + 2)).data, 0x2010);
    assert_eq!(o.bridge.write16(u64::from(w + ireg::MAILBOX_BASE), 0x3412), BUS_OK);
    assert_eq!(o.bridge.read16(u64::from(w + ireg::MAILBOX_BASE)).data, 0x3412);
    // Seen with the swap cleared again, the register holds 0x1234.
    o.set(reg::device(SLOT_ISP0), 0x1100);
    assert_eq!(o.isp_r(SLOT_ISP0, ireg::MAILBOX_BASE), 0x1234);
}

const RQST: u64 = 0x8000;
const RESP: u64 = 0x9000;
const DATA: u64 = 0xa000;

fn inquiry_entry(handle: u32) -> [u8; ENTRY_SIZE] {
    // The 64-byte type 1 request entry, little-endian fields (public Linux
    // qla1280 layout, and qlogic-isp-test.c).
    let mut e = [0u8; ENTRY_SIZE];
    e[0] = ENTRY_TYPE_REQUEST;
    e[1] = 1;
    e[4..8].copy_from_slice(&handle.to_le_bytes());
    e[8..10].copy_from_slice(&(1u16 << 8).to_le_bytes()); // bus 0, target 1, LUN 0
    e[10..12].copy_from_slice(&10u16.to_le_bytes()); // timeout
    e[18..20].copy_from_slice(&1u16.to_le_bytes()); // one data segment
    e[20..26].copy_from_slice(&[0x12, 0, 0, 0, 36, 0]); // INQUIRY, 36 bytes
    e[32..36].copy_from_slice(&(PCI_RAM + DATA as u32).to_le_bytes());
    e[36..40].copy_from_slice(&36u32.to_le_bytes());
    e
}

fn run_inquiry(swap_set: bool) {
    let o = Octane::new();
    let _disk = IspDisk::new("inquiry", &o.isp[0], 1);
    // The rings run on the adapter's thread, from `start` to `stop`.
    o.isp[0].start();
    o.map(SLOT_ISP0, 1, 0x1000_0000, if swap_set { DEV_SWAP_DIRECT } else { 0 });
    o.direct_map_to_heart();
    o.route(0, 18);
    o.isp_w(SLOT_ISP0, ireg::BUS_CTRL, CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE);
    // With the DMA swap clear, RAM holds each little-endian word reversed.
    let reversed = !swap_set;

    assert_eq!(o.isp_queue(SLOT_ISP0, mbox::INIT_REQ_QUEUE, 4, PCI_RAM + RQST as u32), mbox::COMPLETE);
    assert_eq!(o.isp_queue(SLOT_ISP0, mbox::INIT_RSP_QUEUE, 4, PCI_RAM + RESP as u32), mbox::COMPLETE);

    o.place(RQST, &inquiry_entry(0x1234_5678), reversed);
    o.place(DATA, &[0xbb; 40], reversed);
    // The first word of the entry is [type, count, 0, 0]: as the CPU's
    // big-endian word it reads 0x01010000 with the swap clear and 0x00000101
    // with it set.
    let word = u32::from_be_bytes(o.ram_get(RQST, 4).try_into().unwrap());
    assert_eq!(word, if swap_set { 0x0101_0000 } else { 0x0000_0101 });

    o.isp_w(SLOT_ISP0, ireg::REQUEST_QUEUE, 1);
    // The command runs on the adapter's own thread.
    o.isp[0].wait_idle();
    assert_eq!(o.isp_r(SLOT_ISP0, ireg::RESPONSE_QUEUE), 1, "one completion posted");

    let s = o.unplace(RESP, ENTRY_SIZE, reversed);
    assert_eq!(s[0], ENTRY_TYPE_STATUS);
    assert_eq!(u32::from_le_bytes(s[4..8].try_into().unwrap()), 0x1234_5678, "handle");
    assert_eq!(u16::from_le_bytes(s[8..10].try_into().unwrap()), 0, "SCSI GOOD");
    assert_eq!(u16::from_le_bytes(s[10..12].try_into().unwrap()), 0, "complete");

    // The 36 INQUIRY bytes, then the untouched sentinel, in PCI order.
    let d = o.unplace(DATA, 40, reversed);
    assert_eq!(d[0] & 0x1f, 0, "direct access device");
    assert_eq!(&d[8..16], b"SGI     ", "vendor");
    assert_eq!(&d[36..40], &[0xbb; 4], "nothing written past the allocation length");
    // And the raw RAM bytes the driver would load: the vendor id's first word.
    assert_eq!(o.ram_get(DATA + 8, 4), if swap_set { b"SGI ".to_vec() } else { b" IGS".to_vec() });

    // The completion interrupt arrived at HEART as the programmed vector.
    assert_eq!(o.pending(), 1 << 18, "ISP INTA on Bridge pin 0 as vector 18");
    // The driver's acknowledgement drops the line and HEART's bit with it.
    o.isp_w(SLOT_ISP0, ireg::HOST_CMD, 0x7000);
    assert_eq!(o.pending(), 0);
}

#[test]
fn isp_inquiry_through_the_rings_with_the_dma_swap_clear() {
    run_inquiry(false);
}

#[test]
fn isp_inquiry_through_the_rings_with_the_dma_swap_set() {
    run_inquiry(true);
}

// ---- 3. IOC3 Ethernet ----

const TX_DESC: u64 = 0x4000;
const RX_RING: u64 = 0x8000;
const RX_BUF: u64 = 0xc000;

fn frame60() -> Vec<u8> {
    let mut f = vec![0u8; 60];
    f[..6].fill(0xff);
    for (i, b) in f.iter_mut().enumerate().skip(6) {
        *b = i as u8;
    }
    f
}

impl Octane {
    fn eth_setup(&self, swap_set: bool, mac_extra: u32) {
        self.map(SLOT_IOC3, 0, 0x1000_0000, if swap_set { DEV_SWAP_DIRECT } else { 0 });
        self.direct_map_to_heart();
        let rev = swap_set;
        // One receive buffer: the ring entry is a big-endian 64-bit address.
        self.place(RX_RING, &u64::from(PCI_RAM + RX_BUF as u32).to_be_bytes(), rev);
        self.w32(ereg::RX_RING_LO, PCI_RAM + RX_RING as u32);
        self.w32(ereg::RX_TIMER, 0);
        self.w32(ereg::RX_PRODUCE, 8 | ebit::RPIR_ARM);
        self.w32(ereg::INT_ENABLE, ebit::INT_RX_TIMER | ebit::INT_TX_COMPLETE);
        self.w32(
            ereg::MAC_CONTROL,
            ebit::MCR_PROMISC
                | 33 << ebit::MCR_RXOFF_SHIFT
                | ebit::MCR_TX_DMA_EN
                | ebit::MCR_TX_EN
                | ebit::MCR_RX_DMA_EN
                | ebit::MCR_RX_EN
                | mac_extra,
        );
    }

    /// One inline-data transmit descriptor: command and byte count words,
    /// then the frame at +24, all as the IOC3 reads them (big-endian).
    fn eth_send(&self, swap_set: bool, frame: &[u8]) {
        let mut d = [0u8; 128];
        let n = frame.len() as u32;
        d[0..4].copy_from_slice(&(n | ebit::TXD_INT_DONE | ebit::TXD_DATA_VALID).to_be_bytes());
        d[4..8].copy_from_slice(&n.to_be_bytes());
        d[24..24 + frame.len()].copy_from_slice(frame);
        self.place(TX_DESC, &d, swap_set);
        self.w32(ereg::TX_RING_LO, PCI_RAM + TX_DESC as u32);
        self.w32(ereg::TX_PRODUCE, 128);
    }
}

fn run_eth_tx(swap_set: bool) {
    let mut o = Octane::new();
    o.ioc3_map();
    o.eth_setup(swap_set, 0);
    let f = frame60();
    o.eth_send(swap_set, &f);
    assert_eq!(o.sent.pop().unwrap(), f, "the frame went out in wire order");
    assert_eq!(o.i32(ereg::TX_CONSUME), 128 | ebit::TX_CONSUME_IDLE);
    // The descriptor words really are where the swap put them.
    let first = u32::from_be_bytes(o.ram_get(TX_DESC, 4).try_into().unwrap());
    let cmd: u32 = 60 | ebit::TXD_INT_DONE | ebit::TXD_DATA_VALID;
    assert_eq!(first, if swap_set { cmd.swap_bytes() } else { cmd });
    // Frame bytes 8..12 sit at descriptor byte 32, a whole word.
    assert_eq!(o.ram_get(TX_DESC + 32, 4), if swap_set { vec![11, 10, 9, 8] } else { vec![8, 9, 10, 11] });
}

#[test]
fn ioc3_eth_transmits_a_ring_descriptor_and_frame_with_the_dma_swap_clear() {
    run_eth_tx(false);
}

#[test]
fn ioc3_eth_transmits_a_ring_descriptor_and_frame_with_the_dma_swap_set() {
    run_eth_tx(true);
}

fn run_eth_rx(swap_set: bool) {
    let mut o = Octane::new();
    o.ioc3_map();
    o.eth_setup(swap_set, 0);
    let f = frame60();
    o.inject.push(f.clone()).unwrap();
    assert!(o.eth.poll());
    assert_eq!(o.i32(ereg::RX_CONSUME), 8);
    // Receive offset 66: the frame starts at buffer byte 66, in IOC3's own
    // byte order, so a swap-set Bridge reverses each aligned word of it.
    let landed = o.unplace(RX_BUF + 66, f.len(), swap_set);
    assert_eq!(landed, f);
    // Frame bytes 6..10 sit at buffer byte 72, a whole word.
    assert_eq!(o.ram_get(RX_BUF + 72, 4), if swap_set { vec![9, 8, 7, 6] } else { vec![6, 7, 8, 9] });
    // The status header: a valid bit in the first big-endian word.
    let hdr = u32::from_be_bytes(o.unplace(RX_BUF, 4, swap_set).try_into().unwrap());
    assert_ne!(hdr & ebit::RXH_VALID, 0);
}

#[test]
fn ioc3_eth_receive_lands_at_offset_66_with_the_dma_swap_clear() {
    run_eth_rx(false);
}

#[test]
fn ioc3_eth_receive_lands_at_offset_66_with_the_dma_swap_set() {
    run_eth_rx(true);
}

// ---- 4. interrupts at HEART ----

#[test]
fn ioc3_ethernet_interrupt_reaches_heart_as_the_programmed_vector() {
    // octane-test.c, test_octane_ioc3_ethernet_dma_and_interrupt: transmit in
    // MAC loopback, one receive buffer, the Ethernet on the Bridge pin of
    // slot 2's INTA, HEART vector 17.
    let o = Octane::new();
    o.ioc3_map();
    o.route(ENET_PIN, 17);
    o.eth_setup(false, ebit::MCR_LOOPBACK);
    let f = frame60();
    o.eth_send(false, &f);

    assert_eq!(o.i32(ereg::TX_CONSUME), 128 | ebit::TX_CONSUME_IDLE);
    assert_eq!(o.i32(ereg::RX_CONSUME), 8);
    assert_eq!(o.unplace(RX_BUF + 66, 60, false), f);
    let bits = ebit::INT_RX_TIMER | ebit::INT_TX_COMPLETE;
    assert_eq!(o.i32(ereg::INT_STATUS) & bits, bits);
    assert_eq!(o.get(reg::INT_STATUS) & 1 << ENET_PIN, 1 << ENET_PIN);
    assert_eq!(o.pending(), 1 << 17);

    // Writing the status bits back clears the cause, the pin and, with the
    // pin in packet mode, HEART's bit.
    o.w32(ereg::INT_STATUS, bits);
    assert_eq!(o.get(reg::INT_STATUS) & 1 << ENET_PIN, 0);
    assert_eq!(o.pending(), 0);
}

#[test]
fn ioc3_superio_interrupt_arrives_on_bridge_input_4() {
    // octane-test.c, test_serial_a_interrupt_route, with the receive byte
    // supplied by the UART's own loopback.
    let o = Octane::new();
    o.ioc3_map();
    o.route(SUPERIO_PIN, 16);
    o.w8(UART_A + 3, 0x03);
    o.w8(UART_A + 1, 0x01);
    o.w8(UART_A + 4, 0x10);
    o.w8(UART_A, b'R');
    assert_eq!(o.i32(ioc3::reg::INT_STATUS) & ioc3::int::SERIAL_A, ioc3::int::SERIAL_A);
    // The cause is not enabled in IOC3 yet: the pin stays low.
    assert_eq!(o.get(reg::INT_STATUS) & 1 << SUPERIO_PIN, 0);
    assert_eq!(o.pending(), 0);
    o.w32(ioc3::reg::INT_ENABLE_SET, ioc3::int::SERIAL_A);
    assert_eq!(o.get(reg::INT_STATUS) & 1 << SUPERIO_PIN, 1 << SUPERIO_PIN);
    assert_eq!(o.get(reg::INT_STATUS) & 1 << ENET_PIN, 0, "not the Ethernet's pin");
    assert_eq!(o.pending(), 1 << 16);
    // Reading the byte drops the cause, the pin and the HEART bit.
    assert_eq!(o.i8(UART_A), b'R');
    assert_eq!(o.get(reg::INT_STATUS) & 1 << SUPERIO_PIN, 0);
    assert_eq!(o.pending(), 0);
}

// ---- 5. boot flash ----

fn flash_first_words(image: &[u8]) {
    let o = Octane::new();
    let f = Arc::new(RawFlash::new(image, 0x10_0000));
    o.bridge.attach_flash(f, 0x10_0000);
    let w = FLASH_BASE;
    let be = |n: usize| u32::from_be_bytes(image[n..n + 4].try_into().unwrap());
    // The processor's word load at the reset vector (physical 0x1fc0_0000)
    // sees the image's first four bytes, big-endian, whatever the width.
    assert_eq!(o.bridge.read32(u64::from(w)).data, be(0));
    assert_eq!(o.bridge.read32(u64::from(w + 4)).data, be(4));
    assert_eq!(o.bridge.read64(u64::from(w)).data, u64::from(be(0)) << 32 | u64::from(be(4)));
    assert_eq!(o.bridge.read16(u64::from(w)).data, u16::from_be_bytes([image[0], image[1]]));
    assert_eq!(o.bridge.read16(u64::from(w + 2)).data, u16::from_be_bytes([image[2], image[3]]));
    for i in 0..8 {
        assert_eq!(o.bridge.read8(u64::from(w + i)).data, image[i as usize]);
    }
    assert_eq!(0x1f00_0000 + w, 0x1fc0_0000);
}

#[test]
fn flash_serves_a_synthetic_image_big_endian() {
    flash_first_words(&[0x0b, 0xf0, 0x00, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0]);
}

#[test]
fn flash_serves_the_stock_prom_as_the_processor_sees_it() {
    // Skipped without the stock PROM (it is not in the repository).
    let path = std::env::var("IRIS_IP30_PROM").unwrap_or_else(|_| "/inputs/IP30prom.bin".into());
    let Ok(image) = std::fs::read(&path) else {
        eprintln!("skipped: no stock PROM at {path}");
        return;
    };
    // The reset vector's first instruction is a MIPS J: opcode 000010.
    assert_eq!(image[0] >> 2, 0b000010);
    flash_first_words(&image);
    let o = Octane::new();
    o.bridge.attach_flash(Arc::new(RawFlash::new(&image, 0x10_0000)), 0x10_0000);
    assert_eq!(o.bridge.read32(u64::from(FLASH_BASE)).data, 0x0bf0_00f0);
}

/// A device window a BAR decodes whole is published as a run, and the
/// processor's access through it is the access through the Bridge's own
/// window, lane for lane; a window aimed elsewhere drops the run.
#[test]
fn a_bar_run_answers_as_the_window_does() {
    let o = Octane::new();
    let runs = Arc::new(parking_lot::Mutex::new(Vec::<crate::pci::BarRun>::new()));
    let sink = runs.clone();
    o.bridge.on_bar_runs(Box::new(move |r| *sink.lock() = r));
    assert!(runs.lock().is_empty(), "nothing decodes before the PROM places a BAR");
    o.ioc3_map();
    let win = Octane::ioc3_win() as u64;
    let run = runs.lock().iter().find(|r| r.start <= win && win < r.end).cloned().expect("IOC3's window is a run");
    assert_eq!((run.start, run.end), (win, win + (1 << 20)), "the whole 1 MB window");
    let at = win + RTC_DATA as u64;
    o.w8(RTC_INDEX, 0x0e); // a byte of the clock's RAM
    assert_eq!(run.dev.write32(at, 0xaabb_ccdd), BUS_OK);
    assert_eq!(o.i8(RTC_DATA), 0xaa, "a word store through the run lands as through the window");
    o.w8(RTC_DATA + 3, 0x77);
    assert_eq!(run.dev.read8(at).data, o.bridge.read8(at).data);
    assert_eq!(run.dev.read16(at).data, o.bridge.read16(at).data);
    assert_eq!(run.dev.read32(at).data, o.bridge.read32(at).data);
    assert_eq!(run.dev.read64(at).data, o.bridge.read64(at).data);
    assert_eq!(run.dev.read32(at).data, 0x7777_7777);
    // The window aimed at PCI space nothing decodes: no run, and the
    // Bridge's master time-out is what answers there.
    o.set(reg::device(SLOT_IOC3), 0x1000 | (0x3000_0000 >> 20));
    assert!(runs.lock().iter().all(|r| !(r.start <= win && win < r.end)));
}
