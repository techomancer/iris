//! Headless acceptance tests for IP22/IP24 profile wiring, GR2 (XZ/Extreme), and IMPACT stubs.
//!
//! These run in `cargo test --lib` without booting the guest. For live boot
//! checks, point `iris-indigo2-smoke-ci.toml` at a local IRIX root disk (raw or
//! CHD; monitor on 127.0.0.1:8888; CI serial is channel B / IRIX).

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use crate::config::{
        GraphicsBoard, ImpactSection, ImpactSlot, MachineConfig, MachineProfile,
    };
    use crate::eeprom_93c56::Eeprom93c56;
    use crate::ioc::{Ioc, IOC_BASE, IOC_SYS_ID, l1_regs, IOC_INT3_L1_STAT};
    use crate::dev::mgras::{Mgras, GIO_ID, MGRAS_SLOT_GFX_BASE};
    use crate::traits::{BusDevice, Saveable};
    use crate::dev::gr2::{Gr2, Gr2Stats, Gr2Variant, GR2_BASE};

    fn minimal_cfg() -> MachineConfig {
        let mut cfg = MachineConfig::default();
        cfg.scsi.clear();
        cfg
    }

    #[test]
    fn indigo2_smoke_ci_toml_parses_and_validates() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/irix-install/iris-indigo2-smoke-ci.toml"
        );
        let text = std::fs::read_to_string(path).expect("smoke ci toml");
        let cfg: MachineConfig = toml::from_str(&text).expect("parse smoke toml");
        cfg.validate().expect("smoke config validates");
        assert_eq!(cfg.machine.profile, MachineProfile::Indigo2Ip22);
        assert!(cfg.headless);
        assert!(cfg.ci);
    }

    #[test]
    fn impact_solid_on_indigo2_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.impact = ImpactSection {
            gfx: ImpactSlot::Solid,
            exp0: ImpactSlot::None,
            exp1: ImpactSlot::None,
        };
        cfg.validate().expect("Solid IMPACT on Indigo2");
    }

    #[test]
    fn impact_on_indy_rejected() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::IndyIp24;
        cfg.impact.gfx = ImpactSlot::Solid;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("indigo2_ip22"),
            "expected Indigo2-only guard, got: {err}"
        );
    }

    #[test]
    fn xz_on_indy_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::IndyIp24;
        cfg.graphics.board = GraphicsBoard::Xz;
        cfg.graphics.heads = 1;
        cfg.validate().expect("XZ board on Indy");
    }

    #[test]
    fn xz_on_indigo2_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.graphics.board = GraphicsBoard::Xz;
        cfg.validate().expect("XZ board on Indigo2");
    }

    #[test]
    fn extreme_on_indy_rejected() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::IndyIp24;
        cfg.graphics.board = GraphicsBoard::Extreme;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("indigo2_ip22"), "got: {err}");
    }

    #[test]
    fn dual_head_indigo2_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.graphics.heads = 2;
        cfg.validate().expect("dual-head Indigo2");
    }

    #[test]
    fn ioc_sys_id_matches_profile() {
        let sys = IOC_BASE + IOC_SYS_ID;
        let indy = Ioc::new_ci(true);
        assert_eq!(indy.read32(sys).data, 0x26, "Indy guinness sys_id");

        let fullhouse = Ioc::new_ci(false);
        assert_eq!(
            fullhouse.read32(sys).data,
            0x11,
            "Indigo2 fullhouse sys_id"
        );
    }

    #[test]
    fn fullhouse_vblank_routes_sg_retrace_to_l1() {
        let ioc = Ioc::new_ci(false);
        ioc.set_interrupt(crate::ioc::IocInterrupt::VerticalRetrace, true);
        let l1 = ioc.read32(IOC_BASE + IOC_INT3_L1_STAT).data as u8;
        assert_ne!(l1 & l1_regs::VERTICAL_RETRACE, 0, "SG retrace should assert L1 vblank");
        ioc.set_interrupt(crate::ioc::IocInterrupt::VerticalRetrace, false);
        let l1 = ioc.read32(IOC_BASE + IOC_INT3_L1_STAT).data as u8;
        assert_eq!(l1 & l1_regs::VERTICAL_RETRACE, 0);
    }

    #[test]
    fn ioc_fullhouse_gc_select_read_write() {
        let ioc = Ioc::new_ci(false);
        let gc = IOC_BASE + crate::ioc::IOC_GC_SELECT;
        assert_eq!(ioc.write32(gc, 0x0F), crate::traits::BUS_OK);
        assert_eq!(ioc.read32(gc).data, 0x0F, "fullhouse gc_select round-trip");
    }

    fn gr2(variant: Gr2Variant) -> Arc<Gr2> {
        Gr2::new(variant, Gr2Stats {
            heartbeat: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            fasttick: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// What PROM Gr2Probe / Gr2InitInfo read (GR2.h probe section).
    #[test]
    fn gr2_probe_answers() {
        for (variant, rev) in [(Gr2Variant::Xz, 4), (Gr2Variant::Extreme, 6)] {
            let g = gr2(variant);
            assert_eq!(g.read32(GR2_BASE + 0x6a07c).data, 0xdead_beef, "hq.mystery");
            let rd0 = g.read32(GR2_BASE + 0x6c000).data;
            assert_eq!(!rd0 & 0xf, rev, "board rev");
            let rd1 = g.read32(GR2_BASE + 0x6c004).data;
            assert_eq!(rd1 & 0x30, 0x30, "24bpp + Z");
            assert_ne!(!rd1 & 3, 0, "VB rev present");
            // The kernel reads bdvers with lbu at the word address.
            assert_eq!(g.read8(GR2_BASE + 0x6c000).data, rd0 as u8);
            // Config writes must not disturb the ID.
            g.write32(GR2_BASE + 0x6c000, 0x47);
            assert_eq!(g.read32(GR2_BASE + 0x6c000).data, rd0);
        }
    }

    /// GE count probe: installed windows are RAM, the rest do not read back.
    #[test]
    fn gr2_ge_window_probe() {
        for (variant, ges) in [(Gr2Variant::Xz, 2), (Gr2Variant::Extreme, 8)] {
            let g = gr2(variant);
            let mut found = 1;
            for i in 1..8u32 {
                let a = GR2_BASE + 0x68000 + i * 0x400;
                g.write32(a, 0x1f1f_1f1f);
                g.write32(a + 127 * 4, 0x5b5b_5b5b);
                if g.read32(a).data == 0x1f1f_1f1f && g.read32(a + 127 * 4).data == 0x5b5b_5b5b {
                    found = i + 1;
                }
            }
            assert_eq!(found, ges, "{variant:?}");
        }
    }

    #[test]
    fn gr2_save_state_round_trips_shram() {
        let g = gr2(Gr2Variant::Xz);
        g.write32(GR2_BASE + 0x7fff * 4, 1);
        let saved = g.save_state();
        let h = gr2(Gr2Variant::Xz);
        h.load_state(&saved).unwrap();
        assert_eq!(h.read32(GR2_BASE + 0x7fff * 4).data, 1);
    }

    #[test]
    fn mgras_answers_the_gio_id_probe() {
        let cfg = ImpactSection { gfx: ImpactSlot::Solid, exp0: ImpactSlot::None, exp1: ImpactSlot::None };
        let hb = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let ioc = crate::ioc::Ioc::new(false);
        let m = Mgras::new(&cfg, ioc, hb.clone(), hb);
        assert_eq!(m.read32(MGRAS_SLOT_GFX_BASE).data, GIO_ID);
    }

    #[test]
    fn profile_eeprom_independent_of_mc_sysid() {
        // MC SYSID is covered in mc.rs; here we sanity-check both profiles share
        // the same EEPROM path convention (no compile-time indigo2 gate).
        let eeprom = Arc::new(Mutex::new(Eeprom93c56::new()));
        let _indy_mc = crate::mc::MemoryController::new(eeprom.clone(), true, [128, 128, 0, 0]);
        let _i2_mc = crate::mc::MemoryController::new(eeprom, false, [128, 128, 0, 0]);
    }
}
