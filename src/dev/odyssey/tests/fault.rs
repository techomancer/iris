//! GPU residency failures stop processor and display access until reset.

use super::VPRO_WIDGET;
use crate::dev::heart::{host_clock, Heart, HeartConfig};
use crate::dev::heart_mem::HeartRam;
use crate::dev::odyssey::Odyssey;
use crate::display::{Composed, Want};
use crate::traits::{BusDevice, Resettable, BUS_ERR, BUS_OK};
use std::sync::Arc;

fn board() -> Arc<Odyssey> {
    let heart = Arc::new(Heart::new(
        HeartConfig::octane([128 << 20, 0, 0, 0]),
        host_clock(),
    ));
    let ram = Arc::new(HeartRam::try_new([128, 0, 0, 0]).expect("ram"));
    Odyssey::new(
        VPRO_WIDGET as u8,
        "v12",
        heart,
        ram.dma_view(),
        crate::disp::StatusSource::detached(),
        (2, 32),
        (crate::config::RasterUnit::Cpu, "auto"),
    )
    .expect("board")
}

fn identity() -> u64 {
    crate::ip30::widget_large_base(VPRO_WIDGET as u8) + 0x4
}

#[test]
fn gpu_fault_fails_bus_and_scanout_closed_until_reset() {
    let v = board();
    assert_eq!(v.read32(identity()).status, BUS_OK);

    *v.state.shared.gpu_fault.lock() = Some("injected GPU fault".into());
    assert_eq!(v.read32(identity()).status, BUS_ERR);
    assert_eq!(v.write32(identity(), 0), BUS_ERR);

    let mut screen = v.display.screen().lock();
    assert!(matches!(
        v.display.compose(
            &v.state.shared,
            &mut screen,
            Want {
                readback: true,
                gpu: false
            }
        ),
        Composed::NotReady
    ));
    drop(screen);

    v.power_on();
    assert!(v.state.gpu_fault().is_none());
    assert_eq!(v.read32(identity()).status, BUS_OK);
}
