//! Headless acceptance tests for IP22/IP24 profile wiring, XZ, and IMPACT stubs.
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
    use crate::mgras::{self, reg as mgras_reg, Mgras, MGRAS_REG_OFF, MGRAS_SLOT_GFX_BASE};
    use crate::traits::{BusDevice, Saveable};
    use crate::xz::{self, reg as xz_reg, Xz, XZ_REG_BASE};

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
    fn xz_on_indigo2_rejected() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.graphics.board = GraphicsBoard::Xz;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("indy_ip24"), "got: {err}");
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

    #[test]
    fn xz_board_id_probe() {
        let xz = Xz::new();
        let id = xz.read32(XZ_REG_BASE + xz_reg::BOARD_ID).data;
        assert_eq!(id, xz_reg::BOARD_ID_VAL);
        let rev = xz.read32(XZ_REG_BASE + xz_reg::REVISION).data;
        assert_eq!(rev, xz_reg::REVISION_VAL);
        let status = xz.read32(XZ_REG_BASE + xz_reg::STATUS).data;
        assert_eq!(status, xz_reg::STATUS_RESET_VAL);
    }

    #[test]
    fn xz_fifo_write_tracks_depth() {
        let xz = Xz::new();
        assert_eq!(xz.write32(XZ_REG_BASE + xz_reg::FIFO_WRITE, 0x1234_5678), crate::traits::BUS_OK);
        let saved = xz.save_state();
        let depth = saved
            .get("fifo_depth")
            .and_then(|v| v.as_integer())
            .expect("fifo_depth in save_state");
        assert_eq!(depth, 4);
    }

    #[test]
    fn mgras_solid_impact_board_id() {
        let cfg = ImpactSection {
            gfx: ImpactSlot::Solid,
            exp0: ImpactSlot::None,
            exp1: ImpactSlot::None,
        };
        let m = Mgras::new(&cfg);
        let addr = MGRAS_SLOT_GFX_BASE + MGRAS_REG_OFF + mgras_reg::BOARD_ID;
        let id = m.read32(addr).data;
        assert_eq!(id, mgras_reg::board_id_for(ImpactSlot::Solid));
    }

    #[test]
    fn mgras_maximum_three_slot_map() {
        let cfg = ImpactSection {
            gfx: ImpactSlot::Solid,
            exp0: ImpactSlot::High,
            exp1: ImpactSlot::Max,
        };
        let m = Mgras::new(&cfg);
        assert!(m.any_slot());
        for (slot, kind) in [
            (mgras::MGRAS_SLOT_GFX_BASE, ImpactSlot::Solid),
            (mgras::MGRAS_SLOT_EXP0_BASE, ImpactSlot::High),
            (mgras::MGRAS_SLOT_EXP1_BASE, ImpactSlot::Max),
        ] {
            let addr = slot + MGRAS_REG_OFF + mgras_reg::BOARD_ID;
            assert_eq!(m.read32(addr).data, mgras_reg::board_id_for(kind));
        }
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
