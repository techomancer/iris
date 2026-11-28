//! Profile wiring tests for IP22/IP24/IP28 and GR2/IMPACT board selection.
//!
//! These run in `cargo test --lib` without booting the guest. For live boot
//! checks, point `iris-indigo2-smoke-ci.toml` at a local IRIX root disk (raw or
//! CHD; monitor on 127.0.0.1:8888; CI serial is channel B / IRIX).

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::Mutex;

    use crate::config::{
        Cli, GraphicsBoard, MachineConfig, MachineProfile,
    };
    use crate::dev::eeprom_93c56::Eeprom93c56;
    use crate::dev::ioc::{Ioc, IOC_BASE, IOC_SYS_ID, l1_regs, IOC_INT3_L1_STAT};
    use crate::dev::mgras::{Mgras, GIO_ID, MGRAS_SLOT_GFX_BASE};
    use crate::traits::{BusDevice, Saveable};
    use crate::dev::gr2::{Gr2, Gr2Variant, GR2_BASE};

    fn minimal_cfg() -> MachineConfig {
        let mut cfg = MachineConfig::default();
        cfg.devices.clear();
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

    fn virtio_cfg(profile: MachineProfile) -> MachineConfig {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = profile;
        cfg.virtio_serial.enabled = true;
        cfg.resolve_slots().expect("a slot for the card");
        cfg
    }

    fn serial_options(c: &mut MachineConfig) -> &mut crate::config::VirtioSerialConfig {
        c.devices.iter_mut().find_map(|d| match &mut d.device {
            crate::devices::DeviceModel::VirtioSerial(v) => Some(v),
            _ => None,
        }).expect("a virtio_serial entry")
    }

    #[test]
    fn virtio_serial_validates_where_it_has_a_home() {
        for p in [MachineProfile::IndyIp24, MachineProfile::Indigo2Ip22, MachineProfile::Indigo2Ip28, MachineProfile::OctaneIp30] {
            virtio_cfg(p).validate().unwrap_or_else(|e| panic!("{p:?}: {e}"));
        }
    }

    /// The O2, Fuel, Origin 3200 and Onyx 3000 have no spare slot for virtio.
    #[test]
    fn virtio_serial_is_refused_where_the_board_has_no_spare_slot() {
        for p in [MachineProfile::O2Ip32, MachineProfile::FuelIp34, MachineProfile::Origin3200Ip35, MachineProfile::Onyx3000Ip35] {
            let mut cfg = minimal_cfg();
            cfg.machine.profile = p;
            cfg.virtio_serial.enabled = true;
            assert!(cfg.resolve_slots().unwrap_err().contains("virtio"), "{p:?}");
        }
    }

    #[test]
    fn virtio_serial_refuses_a_taken_slot_and_bad_ports() {
        use crate::config::VirtioSerialPort;
        let mut c = MachineConfig::from_toml_str(
            "test_device = true\n[machine]\nprofile = \"indigo2_ip28\"\n[virtio_serial]\nenabled = true\nslot = \"exp0\"",
        ).unwrap();
        assert!(c.resolve_slots().unwrap_err().contains("gio0 is taken by the test_device"));

        let mut c = virtio_cfg(MachineProfile::IndyIp24);
        let port = |n: &str| VirtioSerialPort { name: Some(n.into()), ..Default::default() };
        serial_options(&mut c).ports = vec![port("a"), port("a")];
        assert!(c.validate().unwrap_err().contains("used twice"));
        serial_options(&mut c).ports = (0..32).map(|i| port(&format!("p{i}"))).collect();
        assert!(c.validate().unwrap_err().contains("at most 31"));
        serial_options(&mut c).ports = vec![VirtioSerialPort { tcp_port: Some(1), socket: Some("/x".into()), ..Default::default() }];
        assert!(c.validate().unwrap_err().contains("exclusive"));
    }

    #[test]
    fn virtio_serial_section_parses() {
        let cfg: MachineConfig = toml::from_str(
            r#"
            [virtio_serial]
            enabled = true
            slot = "exp0"
            [[virtio_serial.port]]
            console = true
            tcp_port = 8890
            [[virtio_serial.port]]
            name = "org.iris.agent.0"
            socket = "/tmp/iris-agent.sock"
            "#,
        )
        .expect("parses");
        assert_eq!(cfg.virtio_serial.ports.len(), 2);
        assert_eq!(cfg.virtio_serial.ports[1].name.as_deref(), Some("org.iris.agent.0"));
        // Unset, it stays out of a saved config.
        assert!(!toml::to_string(&MachineConfig::default()).unwrap().contains("virtio_serial"));
    }

    #[test]
    fn impact_solid_on_indigo2_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::SolidImpact)));
        cfg.validate().expect("Solid IMPACT on Indigo2");
    }

    #[test]
    fn impact_on_indy_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::IndyIp24;
        cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::SolidImpact)));
        cfg.validate().expect("Solid IMPACT on Indy");
    }

    #[test]
    fn newport_on_ip28_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::Indigo2Ip28;
        cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::Newport)));
        cfg.validate().expect("Newport on IP28");
    }

    #[test]
    fn xz_on_ip28_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::Indigo2Ip28;
        cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::Xz)));
        cfg.validate().expect("XZ on IP28");
    }

    #[test]
    fn impact_models_parse_short_or_long() {
        for (names, board) in [(["solid", "solidimpact"], GraphicsBoard::SolidImpact), (["high", "highimpact"], GraphicsBoard::HighImpact), (["max", "maximpact"], GraphicsBoard::MaxImpact)] {
            for name in names {
                assert_eq!(name.parse::<GraphicsBoard>().unwrap(), board, "{name}");
            }
            assert_eq!(board.as_str(), names[0], "the short form is written");
        }
    }

    #[test]
    fn the_old_graphics_flag_still_parses_so_it_can_be_refused() {
        use clap::Parser;
        let cli = Cli::try_parse_from(["iris", "--graphics", "impact:high"]).unwrap();
        assert_eq!(cli.graphics.as_deref(), Some("impact:high"));
    }

    #[test]
    fn xz_on_indy_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::IndyIp24;
        cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::Xz)));
        cfg.validate().expect("XZ board on Indy");
    }

    #[test]
    fn xz_on_indigo2_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::Xz)));
        cfg.validate().expect("XZ board on Indigo2");
    }

    #[test]
    fn extreme_on_indy_rejected() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::IndyIp24;
        cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::Extreme)));
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("no free slot for a graphics board"), "got: {err}");
    }

    #[test]
    fn dual_head_indigo2_validates() {
        let mut cfg = minimal_cfg();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.set_dual_head(true);
        cfg.validate().expect("dual-head Indigo2");
    }

    #[test]
    fn ioc_sys_id_matches_profile() {
        let sys = (IOC_BASE + IOC_SYS_ID) as u64;
        let indy = Ioc::new(&Arc::new(crate::hptimer::TimerManager::new()), true);
        assert_eq!(indy.read32(sys).data, 0x26, "Indy guinness sys_id");

        let fullhouse = Ioc::new(&Arc::new(crate::hptimer::TimerManager::new()), false);
        assert_eq!(
            fullhouse.read32(sys).data,
            0x11,
            "Indigo2 fullhouse sys_id"
        );
    }

    #[test]
    fn fullhouse_vblank_routes_sg_retrace_to_l1() {
        let ioc = Ioc::new(&Arc::new(crate::hptimer::TimerManager::new()), false);
        ioc.set_interrupt(crate::dev::ioc::IocInterrupt::VerticalRetrace, true);
        let l1 = ioc.read32((IOC_BASE + IOC_INT3_L1_STAT) as u64).data as u8;
        assert_ne!(l1 & l1_regs::VERTICAL_RETRACE, 0, "SG retrace should assert L1 vblank");
        ioc.set_interrupt(crate::dev::ioc::IocInterrupt::VerticalRetrace, false);
        let l1 = ioc.read32((IOC_BASE + IOC_INT3_L1_STAT) as u64).data as u8;
        assert_eq!(l1 & l1_regs::VERTICAL_RETRACE, 0);
    }

    #[test]
    fn ioc_fullhouse_gc_select_read_write() {
        let ioc = Ioc::new(&Arc::new(crate::hptimer::TimerManager::new()), false);
        let gc = (IOC_BASE + crate::dev::ioc::IOC_GC_SELECT) as u64;
        assert_eq!(ioc.write32(gc, 0x0F), crate::traits::BUS_OK);
        assert_eq!(ioc.read32(gc).data, 0x0F, "fullhouse gc_select round-trip");
    }

    fn gr2(variant: Gr2Variant) -> Arc<Gr2> {
        Gr2::new(variant, crate::status::MachineStatus::detached())
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
                let a = GR2_BASE + 0x68000 + (i * 0x400) as u64;
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
        let m = Mgras::new(GraphicsBoard::SolidImpact, std::array::from_fn(|_| crate::irq::IrqLine::unconnected()), crate::status::MachineStatus::detached());
        assert_eq!(m.read32(MGRAS_SLOT_GFX_BASE as u64).data, GIO_ID);
    }

    #[test]
    fn profile_eeprom_independent_of_mc_sysid() {
        // MC SYSID is covered in mc.rs; here we sanity-check both profiles share
        // the same EEPROM path convention (no compile-time indigo2 gate).
        let eeprom = Arc::new(Mutex::new(Eeprom93c56::new()));
        let _indy_mc = crate::dev::mc::MemoryController::new(eeprom.clone(), true, [128, 128, 0, 0]);
        let _i2_mc = crate::dev::mc::MemoryController::new(eeprom, false, [128, 128, 0, 0]);
    }
}
