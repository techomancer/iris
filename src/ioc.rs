use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::devlog::LogModule;
use std::sync::mpsc;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, Resettable, Saveable, MachineEvent};
use crate::snapshot::{get_field, toml_u8, hex_u8};
use crate::z85c30::{Z85c30, IrqCallback};
use crate::pit8254::{Pit8254, TimerCallback};
use crate::mips_core::{CAUSE_IP2, CAUSE_IP3, CAUSE_IP4, CAUSE_IP5, CAUSE_IP6};
use crate::ps2::{Ps2Controller, Ps2Callback};
use crate::hptimer::TimerManager;
use std::io::Write;

pub const IOC_BASE: u32 = 0x1FBD9800;
pub const IOC_SIZE: u32 = 0x100;

// Register Offsets
pub const IOC_PL_DATA: u32 = 0x00;
pub const IOC_PL_CNTL: u32 = 0x04;
pub const IOC_PL_STAT: u32 = 0x08;
pub const IOC_PL_DMA_CNTL: u32 = 0x0C;
pub const IOC_PL_INT_STAT: u32 = 0x10;
pub const IOC_PL_INT_MASK: u32 = 0x14;
pub const IOC_PL_TIMER1: u32 = 0x18;
pub const IOC_PL_TIMER2: u32 = 0x1C;
pub const IOC_PL_TIMER3: u32 = 0x20;
pub const IOC_PL_TIMER4: u32 = 0x24;

pub const IOC_SERIAL1_CMD: u32 = 0x30;
pub const IOC_SERIAL1_DATA: u32 = 0x34;
pub const IOC_SERIAL2_CMD: u32 = 0x38;
pub const IOC_SERIAL2_DATA: u32 = 0x3C;

pub const IOC_KBD_MOUSE_DATA: u32 = 0x40;
pub const IOC_KBD_MOUSE_CMD: u32 = 0x44;
pub const IOC_GC_SELECT: u32 = 0x48;
pub const IOC_GEN_CNTL: u32 = 0x4C;
pub const IOC_PANEL: u32 = 0x50;
pub const IOC_SYS_ID: u32 = 0x58;
pub const IOC_READ: u32 = 0x60;
pub const IOC_DMA_SEL: u32 = 0x68;
pub const IOC_RESET: u32 = 0x70;
pub const IOC_WRITE: u32 = 0x78;

pub const IOC_INT3_L0_STAT: u32 = 0x80;
pub const IOC_INT3_L0_MASK: u32 = 0x84;
pub const IOC_INT3_L1_STAT: u32 = 0x88;
pub const IOC_INT3_L1_MASK: u32 = 0x8C;
pub const IOC_INT3_MAP_STAT: u32 = 0x90;
pub const IOC_INT3_MAP_MASK0: u32 = 0x94;
pub const IOC_INT3_MAP_MASK1: u32 = 0x98;
pub const IOC_INT3_MAP_POL: u32 = 0x9C;
pub const IOC_INT3_TMR_CLR: u32 = 0xA0;
pub const IOC_INT3_ERR_STAT: u32 = 0xA4;

pub const IOC_TIMER_CNT0: u32 = 0xB0;
pub const IOC_TIMER_CNT1: u32 = 0xB4;
pub const IOC_TIMER_CNT2: u32 = 0xB8;
pub const IOC_TIMER_CTL: u32 = 0xBC;

pub mod l0_regs {
    pub const MAP_INT0: u8 = 1 << 7;
    pub const GRAPHICS: u8 = 1 << 6;
    pub const PARALLEL: u8 = 1 << 5;
    pub const MC_DMA: u8 = 1 << 4;
    pub const ETHERNET: u8 = 1 << 3;
    pub const SCSI1: u8 = 1 << 2;
    pub const SCSI0: u8 = 1 << 1;
    pub const FIFO_FULL: u8 = 1 << 0;
}

pub mod l1_regs {
    pub const VERTICAL_RETRACE: u8 = 1 << 7;
    pub const VIDEO_VSYNC: u8 = 1 << 6;
    pub const AC_FAIL: u8 = 1 << 5;
    pub const HPC_DMA: u8 = 1 << 4;
    pub const MAP_INT1: u8 = 1 << 3;
    pub const GP2: u8 = 1 << 2;
    pub const PANEL: u8 = 1 << 1;
    pub const GP0: u8 = 1 << 0;
}

pub mod map_regs {
    pub const SERIAL: u8 = 1 << 5;
    pub const KBD_MOUSE: u8 = 1 << 4;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IocInterrupt {
    // Local 0 Sources
    Graphics,
    Parallel,
    McDma,
    Ethernet,
    Scsi1,
    Scsi0,
    FifoFull,

    // Local 1 Sources
    VerticalRetrace,
    VideoVsync,
    AcFail,
    HpcDma,
    Gp2,
    Panel,
    Gp0,

    // Mappable Sources
    Serial,
    KbMouse,
    Mappable0, // Timer 0
    Mappable1, // Timer 1
    Mappable2,
    Mappable3,
    Mappable6,
    Mappable7,
}

struct IocState {
    sys_id: u8,
    
    // INT3 Registers
    l0_stat: u8,
    l0_mask: u8,
    l1_stat: u8,
    l1_mask: u8,
    map_stat: u8,
    map_mask0: u8,
    map_mask1: u8,
    map_pol: u8,
    err_stat: u8,

    // Misc Registers
    gc_select: u8,
    gen_cntl: u8,
    panel: u8,
    read_reg: u8,
    dma_sel: u8,
    reset_reg: u8,
    write_reg: u8,
    interrupts: Option<Arc<AtomicU64>>,
}

struct IocIrqLine {
    state: Arc<Mutex<IocState>>,
    source: IocInterrupt,
}

struct IocTimerCallback {
    state: Arc<Mutex<IocState>>,
    source: IocInterrupt,
}

impl TimerCallback for IocTimerCallback {
    fn callback(&self) {
        let mut state = self.state.lock();
        match self.source {
            IocInterrupt::Mappable0 => state.map_stat |= 1 << 0,
            IocInterrupt::Mappable1 => state.map_stat |= 1 << 1,
            _ => {}
        }
        state.update_interrupts();
    }
}

impl IrqCallback for IocIrqLine {
    fn set_level(&self, level: bool) {
        let mut state = self.state.lock();
        match self.source {
            IocInterrupt::Serial => if level { state.map_stat |= map_regs::SERIAL } else { state.map_stat &= !map_regs::SERIAL },
            IocInterrupt::KbMouse => if level { state.map_stat |= map_regs::KBD_MOUSE } else { state.map_stat &= !map_regs::KBD_MOUSE },
            _ => {} // Only Serial supported via this callback for now
        }
        state.update_interrupts();
    }
}

impl Ps2Callback for IocIrqLine {
    fn set_interrupt(&self, active: bool) {
        self.set_level(active);
    }
}

#[derive(Clone)]
pub struct Ioc {
    state: Arc<Mutex<IocState>>,
    scc: Z85c30,
    pit: Pit8254,
    ps2: Arc<Ps2Controller>,
    guinness: bool,
    /// Sender for async machine events (power-off).
    event_tx: Arc<std::sync::OnceLock<mpsc::SyncSender<MachineEvent>>>,
    /// Shared heartbeat — IOC sets/clears HB_LED_RED/GREEN bits directly.
    heartbeat: Arc<std::sync::OnceLock<Arc<AtomicU64>>>,
    /// Shared timer manager for PIT channels.
    timer_manager: Arc<std::sync::OnceLock<Arc<TimerManager>>>,
}

impl Ioc {
    pub fn new(guinness: bool) -> Self {
        let sys_id = if guinness { 0x26 } else { 0x11 }; // primarily prom looks at bit 1 to detect full house.
        let state = Arc::new(Mutex::new(IocState {
            sys_id,
            l0_stat: 0,
            l0_mask: 0,
            l1_stat: 0,
            l1_mask: 0,
            map_stat: 0,
            map_mask0: 0,
            map_mask1: 0,
            map_pol: 0,
            err_stat: 0,
            gc_select: 0,
            gen_cntl: 0,
            panel: 1, // Power State (Bit 0) = 1 (On)
            read_reg: 0x70, // Ethernet/SCSI Power Good (Bits 6,5,4 = 1)
            dma_sel: 0,
            reset_reg: 0,
            write_reg: 0,
            interrupts: None,
        }));

        let serial_irq = Arc::new(IocIrqLine {
            state: state.clone(),
            source: IocInterrupt::Serial,
        });

        let timer0_cb = Arc::new(IocTimerCallback {
            state: state.clone(),
            source: IocInterrupt::Mappable0,
        });

        let timer1_cb = Arc::new(IocTimerCallback {
            state: state.clone(),
            source: IocInterrupt::Mappable1,
        });

        let ps2_cb = Arc::new(IocIrqLine {
            state: state.clone(),
            source: IocInterrupt::KbMouse,
        });

        Self {
            state,
            scc: Z85c30::new(Some(serial_irq)),
            pit: Pit8254::new(1_000_000, Some(timer0_cb), Some(timer1_cb), None),
            ps2: Arc::new(Ps2Controller::new(Some(ps2_cb))),
            guinness,
            event_tx: Arc::new(std::sync::OnceLock::new()),
            heartbeat: Arc::new(std::sync::OnceLock::new()),
            timer_manager: Arc::new(std::sync::OnceLock::new()),
        }
    }

    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timer_manager.set(tm.clone());
        self.pit.set_timer_manager(tm);
    }

    pub fn set_event_sender(&self, tx: mpsc::SyncSender<MachineEvent>) {
        let _ = self.event_tx.set(tx);
    }

    pub fn set_heartbeat(&self, heartbeat: Arc<AtomicU64>) {
        let _ = self.heartbeat.set(heartbeat);
    }

    pub fn set_interrupts(&self, interrupts: Arc<AtomicU64>) {
        self.state.lock().interrupts = Some(interrupts);
    }

    pub fn set_interrupt(&self, source: IocInterrupt, active: bool) {
        // Note: map_pol (polarity) register is currently ignored.
        // We assume active-high logic internally for now.
        match source {
            IocInterrupt::VerticalRetrace | IocInterrupt::VideoVsync => {},
            _ => dlog_dev!(LogModule::Ioc, "IOC: Set Interrupt {:?} = {}", source, active),
        }
        let mut state = self.state.lock();
        match source {
            // Local 0
            IocInterrupt::Graphics => if active { state.l0_stat |= l0_regs::GRAPHICS } else { state.l0_stat &= !l0_regs::GRAPHICS },
            IocInterrupt::Parallel => if active { state.l0_stat |= l0_regs::PARALLEL } else { state.l0_stat &= !l0_regs::PARALLEL },
            IocInterrupt::McDma => if active { state.l0_stat |= l0_regs::MC_DMA } else { state.l0_stat &= !l0_regs::MC_DMA },
            IocInterrupt::Ethernet => if active { state.l0_stat |= l0_regs::ETHERNET } else { state.l0_stat &= !l0_regs::ETHERNET },
            IocInterrupt::Scsi1 => if active { state.l0_stat |= l0_regs::SCSI1 } else { state.l0_stat &= !l0_regs::SCSI1 },
            IocInterrupt::Scsi0 => if active { state.l0_stat |= l0_regs::SCSI0 } else { state.l0_stat &= !l0_regs::SCSI0 },
            IocInterrupt::FifoFull => if active { state.l0_stat |= l0_regs::FIFO_FULL } else { state.l0_stat &= !l0_regs::FIFO_FULL },

            // Local 1
            IocInterrupt::VerticalRetrace => if active { state.l1_stat |= l1_regs::VERTICAL_RETRACE } else { state.l1_stat &= !l1_regs::VERTICAL_RETRACE },
            IocInterrupt::VideoVsync => if active { state.l1_stat |= l1_regs::VIDEO_VSYNC } else { state.l1_stat &= !l1_regs::VIDEO_VSYNC },
            IocInterrupt::AcFail => if active { state.l1_stat |= l1_regs::AC_FAIL } else { state.l1_stat &= !l1_regs::AC_FAIL },
            IocInterrupt::HpcDma => if active { state.l1_stat |= l1_regs::HPC_DMA } else { state.l1_stat &= !l1_regs::HPC_DMA },
            IocInterrupt::Gp2 => if active { state.l1_stat |= l1_regs::GP2 } else { state.l1_stat &= !l1_regs::GP2 },
            IocInterrupt::Panel => if active { state.l1_stat |= l1_regs::PANEL } else { state.l1_stat &= !l1_regs::PANEL },
            IocInterrupt::Gp0 => if active { state.l1_stat |= l1_regs::GP0 } else { state.l1_stat &= !l1_regs::GP0 },

            // Mappable
            IocInterrupt::Serial => if active { state.map_stat |= map_regs::SERIAL } else { state.map_stat &= !map_regs::SERIAL },
            IocInterrupt::KbMouse => if active { state.map_stat |= map_regs::KBD_MOUSE } else { state.map_stat &= !map_regs::KBD_MOUSE },
            IocInterrupt::Mappable0 => if active { state.map_stat |= 1 << 0 } else { state.map_stat &= !(1 << 0) },
            IocInterrupt::Mappable1 => if active { state.map_stat |= 1 << 1 } else { state.map_stat &= !(1 << 1) },
            IocInterrupt::Mappable2 => if active { state.map_stat |= 1 << 2 } else { state.map_stat &= !(1 << 2) },
            IocInterrupt::Mappable3 => if active { state.map_stat |= 1 << 3 } else { state.map_stat &= !(1 << 3) },
            IocInterrupt::Mappable6 => if active { state.map_stat |= 1 << 6 } else { state.map_stat &= !(1 << 6) },
            IocInterrupt::Mappable7 => if active { state.map_stat |= 1 << 7 } else { state.map_stat &= !(1 << 7) },
        }
        state.update_interrupts();
    }

    pub fn ps2(&self) -> Arc<Ps2Controller> {
        self.ps2.clone()
    }

    pub fn scc(&self) -> &Z85c30 {
        &self.scc
    }

    pub fn pit(&self) -> &Pit8254 {
        &self.pit
    }

    pub fn register_locks(&self) {
        use crate::locks::register_lock_fn;
        let s = self.state.clone();
        register_lock_fn("ioc::state", move || s.is_locked());
        // SCC (Z85c30) channels
        self.scc.register_locks();
        // PS/2
        let ps2 = self.ps2.clone();
        register_lock_fn("ps2::state", move || ps2.is_state_locked());
    }
}

impl Device for Ioc {
    fn step(&self, _cycles: u64) {
        // TODO: Implement timer stepping
    }

    fn stop(&self) { self.scc.stop(); self.pit.stop(); self.ps2.stop(); }
    fn start(&self) {
        dlog_dev!(LogModule::Ioc, "IOC: start() called");
        self.scc.start();
        self.pit.start();
        self.ps2.start();
    }
    fn is_running(&self) -> bool { self.scc.is_running() }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        let mut cmds = vec![("ioc".to_string(), "IOC commands: ioc status".to_string())];
        cmds.extend(self.scc.register_commands());
        cmds.extend(self.pit.register_commands());
        cmds.extend(self.ps2.register_commands());
        cmds
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd == "ioc" {
            return Err("Usage: ioc status".to_string());
        }
        if cmd == "serial" {
            return self.scc.execute_command(cmd, args, writer);
        }
        if cmd == "pit" {
            return self.pit.execute_command(cmd, args, writer);
        }
        if cmd == "ps2" {
            return self.ps2.execute_command(cmd, args, writer);
        }
        Err("Command not found".to_string())
    }
}

impl BusDevice for Ioc {
    fn read8(&self, addr: u32) -> BusRead8 {
        let offset = (addr - IOC_BASE) & !3;

        // Lock state only for IOC registers, not for SCC/PIT passthrough
        // This prevents deadlock when SCC callback tries to lock state

        // Serial ports (SCC) - direct 8-bit access
        if offset >= IOC_SERIAL1_CMD && offset <= IOC_SERIAL2_DATA {
            let idx = (offset - IOC_SERIAL1_CMD) >> 2;
            return self.scc.read(idx);
        }

        // Timers (PIT) - direct 8-bit access
        if offset >= IOC_TIMER_CNT0 && offset <= IOC_TIMER_CTL + 3 {
            let idx = (offset - IOC_TIMER_CNT0) >> 2;
            dlog_dev!(LogModule::Ioc, "IOC: Read PIT channel {} (offset {:02x})", idx, offset);
            return self.pit.read(idx);
        }

        // PS/2 Keyboard/Mouse - direct access to avoid lock inversion
        if offset == IOC_KBD_MOUSE_DATA {
            return BusRead8::ok(self.ps2.read_data());
        }
        if offset == IOC_KBD_MOUSE_CMD {
            return BusRead8::ok(self.ps2.read_status());
        }

        // IOC registers - all 8-bit
        let state = self.state.lock();

        let val = match offset {
            IOC_SYS_ID => state.sys_id,

            IOC_PL_DATA => 0,
            IOC_PL_CNTL => 0,
            IOC_PL_STAT => 0,

            IOC_INT3_L0_STAT => state.l0_stat,
            IOC_INT3_L0_MASK => state.l0_mask,
            IOC_INT3_L1_STAT => state.l1_stat,
            IOC_INT3_L1_MASK => state.l1_mask,
            IOC_INT3_MAP_STAT => state.map_stat,
            IOC_INT3_MAP_MASK0 => state.map_mask0,
            IOC_INT3_MAP_MASK1 => state.map_mask1,
            IOC_INT3_MAP_POL => state.map_pol,
            IOC_INT3_ERR_STAT => state.err_stat,

            IOC_GC_SELECT => state.gc_select,
            IOC_GEN_CNTL => state.gen_cntl,
            IOC_PANEL => state.panel,
            IOC_READ => state.read_reg,
            IOC_DMA_SEL => state.dma_sel,
            IOC_RESET => state.reset_reg,
            IOC_WRITE => state.write_reg,

            _ => {
                dlog_dev!(LogModule::Ioc, "IOC: Read8 offset {:02x}", offset);
                0
            }
        };
        BusRead8::ok(val)
    }

    fn write8(&self, addr: u32, val: u8) -> u32 {
        let offset = (addr - IOC_BASE) & !3;

        // Serial ports (SCC) - direct 8-bit access
        if offset >= IOC_SERIAL1_CMD && offset <= IOC_SERIAL2_DATA {
            let idx = (offset - IOC_SERIAL1_CMD) >> 2;
            return self.scc.write(idx, val);
        }

        // Timers (PIT) - direct 8-bit access
        if offset >= IOC_TIMER_CNT0 && offset <= IOC_TIMER_CTL + 3 {
            let idx = (offset - IOC_TIMER_CNT0) >> 2;
            dlog_dev!(LogModule::Ioc, "IOC: Write PIT channel {} (offset {:02x}) val {:02x}", idx, offset, val);
            return self.pit.write(idx, val);
        }

        // PS/2 Keyboard/Mouse - direct access to avoid lock inversion
        if offset == IOC_KBD_MOUSE_DATA {
            self.ps2.write_data(val);
            return BUS_OK;
        }
        if offset == IOC_KBD_MOUSE_CMD {
            self.ps2.write_command(val);
            return BUS_OK;
        }

        let mut state = self.state.lock();

        match offset {
            IOC_PL_DATA => { dlog_dev!(LogModule::Ioc, "IOC: Write PL_DATA val {:02x}", val); },
            IOC_PL_CNTL => { dlog_dev!(LogModule::Ioc, "IOC: Write PL_CNTL val {:02x}", val); },

            IOC_INT3_L0_MASK => state.l0_mask = val,
            IOC_INT3_L1_MASK => state.l1_mask = val,
            IOC_INT3_MAP_MASK0 => state.map_mask0 = val,
            IOC_INT3_MAP_MASK1 => state.map_mask1 = val,
            IOC_INT3_MAP_POL => state.map_pol = val,
            IOC_INT3_TMR_CLR => {
                dlog_dev!(LogModule::Ioc, "IOC: Timer Clear val {:02x}", val);
                state.map_stat &= !(val & 0x3);
            }

            IOC_GC_SELECT => state.gc_select = val,
            IOC_GEN_CNTL => state.gen_cntl = val,
            IOC_PANEL => {
                // Bits 6, 4, 1 are W1C (Write 1 to Clear)
                let mut current = state.panel;
                if (val & (1 << 6)) != 0 { current &= !(1 << 6); }
                if (val & (1 << 4)) != 0 { current &= !(1 << 4); }
                if (val & (1 << 1)) != 0 { current &= !(1 << 1); }
                // Bit 0 is RW (Power State, active low: 0 = off)
                let was_on = (current & 1) != 0;
                current = (current & !1) | (val & 1);
                state.panel = current;
                let now_off = (current & 1) == 0;
                if was_on && now_off {
                    if let Some(tx) = self.event_tx.get() {
                        dlog_dev!(LogModule::Ioc, "IOC: front panel power-off requested");
                        let _ = tx.try_send(MachineEvent::PowerOff);
                    }
                }
            }
            IOC_DMA_SEL => state.dma_sel = val,
            IOC_RESET => {
                use crate::rex3::Rex3;
                state.reset_reg = val;

                // LED bits are active-low: 0x10=LED_RED_OFF, 0x20=LED_GREEN_OFF
                // bit SET = LED off, bit CLEAR = LED on — update heartbeat unconditionally
                if let Some(hb) = self.heartbeat.get() {
                    if (val & 0x10) == 0 {
                        hb.fetch_or(Rex3::HB_LED_RED, Ordering::Relaxed);
                    } else {
                        hb.fetch_and(!Rex3::HB_LED_RED, Ordering::Relaxed);
                    }
                    if (val & 0x20) == 0 {
                        hb.fetch_or(Rex3::HB_LED_GREEN, Ordering::Relaxed);
                    } else {
                        hb.fetch_and(!Rex3::HB_LED_GREEN, Ordering::Relaxed);
                    }
                }
            },
            IOC_WRITE => state.write_reg = val,

            _ => {
                dlog_dev!(LogModule::Ioc, "IOC: Write8 offset {:02x} val {:02x}", offset, val);
            }
        }
        // Update interrupts after any write that might affect masks or status
        state.update_interrupts();
        BUS_OK
    }

    fn read32(&self, addr: u32) -> BusRead32 {
        //println!("IOC: Read32 addr {:08x}", addr);
        // IOC registers are accessed as 32-bit words with data in low 8 bits
        // Address should be word-aligned
        let aligned_addr = addr & !3;
        let r = self.read8(aligned_addr);
        if r.is_ok() { BusRead32::ok(r.data as u32) } else { BusRead32 { status: r.status, data: 0 } }
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        //println!("IOC: Write32 addr {:08x} val {:08x}", addr, val);
        // IOC registers are accessed as 32-bit words with data in low 8 bits
        // Address should be word-aligned
        let aligned_addr = addr & !3;
        // Extract low 8 bits (bits 7:0)
        let val8 = (val & 0xFF) as u8;
        self.write8(aligned_addr, val8)
    }
}

impl IocState {
    fn update_interrupts(&mut self) {
        // 1. Update Mappable Interrupts (MAP_INT0, MAP_INT1)
        let map_int0 = (self.map_stat & self.map_mask0) != 0;
        let map_int1 = (self.map_stat & self.map_mask1) != 0;

        // Update Local 0 Status Bit 7
        if map_int0 {
            self.l0_stat |= l0_regs::MAP_INT0;
        } else {
            self.l0_stat &= !l0_regs::MAP_INT0;
        }

        // Update Local 1 Status Bit 3
        if map_int1 {
            self.l1_stat |= l1_regs::MAP_INT1;
        } else {
            self.l1_stat &= !l1_regs::MAP_INT1;
        }

        // 2. Calculate CPU Interrupts
        // Local 0 -> IP2 (MIPS Int 0)
        let ip2 = (self.l0_stat & self.l0_mask) != 0;
        
        // Local 1 -> IP3 (MIPS Int 1)
        let ip3 = (self.l1_stat & self.l1_mask) != 0;

        // Level 2: Timer 0 -> IP4
        // Timer 0 is bit 0 of map_stat (latched)
        let ip4 = (self.map_stat & 0x01) != 0;

        // Level 3: Timer 1 -> IP5
        // Timer 1 is bit 1 of map_stat (latched)
        let ip5 = (self.map_stat & 0x02) != 0;

        // Level 4: Bus Error -> IP6
        let ip6 = self.err_stat != 0;

        // 3. Signal CPU
        if let Some(interrupts) = &self.interrupts {
            let mut set_mask = 0;
            let mut clear_mask = 0;

            if ip2 {
                set_mask |= CAUSE_IP2 as u64;
            } else {
                clear_mask |= CAUSE_IP2 as u64;
            }
            if ip3 {
                set_mask |= CAUSE_IP3 as u64;
            } else {
                clear_mask |= CAUSE_IP3 as u64;
            }
            if ip4 { set_mask |= CAUSE_IP4 as u64; } else { clear_mask |= CAUSE_IP4 as u64; }
            if ip5 { set_mask |= CAUSE_IP5 as u64; } else { clear_mask |= CAUSE_IP5 as u64; }
            if ip6 { set_mask |= CAUSE_IP6 as u64; } else { clear_mask |= CAUSE_IP6 as u64; }

            if set_mask != 0 {
                interrupts.fetch_or(set_mask, Ordering::SeqCst);
            }
            if clear_mask != 0 {
                interrupts.fetch_and(!clear_mask, Ordering::SeqCst);
            }
        }
    }
}
// ============================================================================
// Resettable + Saveable for Ioc
// ============================================================================

impl Resettable for Ioc {
    fn power_on(&self) {
        let mut state = self.state.lock();
        state.l0_stat = 0;
        state.l0_mask = 0;
        state.l1_stat = 0;
        state.l1_mask = 0;
        state.map_stat = 0;
        state.map_mask0 = 0;
        state.map_mask1 = 0;
        state.map_pol = 0;
        state.err_stat = 0;
        state.gc_select = 0;
        state.gen_cntl = 0;
        state.panel = 1;        // power-on: power state bit = 1
        state.read_reg = 0x70;  // ethernet/SCSI power good
        state.dma_sel = 0;
        state.reset_reg = 0;
        state.write_reg = 0;
        // Clear CPU interrupt lines
        if let Some(irqs) = &state.interrupts {
            use std::sync::atomic::Ordering;
            irqs.store(0, Ordering::SeqCst);
        }
    }
}

impl Saveable for Ioc {
    fn save_state(&self) -> toml::Value {
        let state = self.state.lock();
        let mut tbl = toml::map::Map::new();
        macro_rules! u8f { ($f:ident) => { tbl.insert(stringify!($f).into(), hex_u8(state.$f)); } }
        u8f!(l0_stat); u8f!(l0_mask); u8f!(l1_stat); u8f!(l1_mask);
        u8f!(map_stat); u8f!(map_mask0); u8f!(map_mask1); u8f!(map_pol); u8f!(err_stat);
        u8f!(gc_select); u8f!(gen_cntl); u8f!(panel); u8f!(read_reg);
        u8f!(dma_sel); u8f!(reset_reg); u8f!(write_reg);
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut state = self.state.lock();
        macro_rules! ldu8 { ($f:ident) => {
            if let Some(x) = get_field(v, stringify!($f)) { state.$f = toml_u8(x).unwrap_or(state.$f); }
        }}
        ldu8!(l0_stat); ldu8!(l0_mask); ldu8!(l1_stat); ldu8!(l1_mask);
        ldu8!(map_stat); ldu8!(map_mask0); ldu8!(map_mask1); ldu8!(map_pol); ldu8!(err_stat);
        ldu8!(gc_select); ldu8!(gen_cntl); ldu8!(panel); ldu8!(read_reg);
        ldu8!(dma_sel); ldu8!(reset_reg); ldu8!(write_reg);
        state.update_interrupts();
        Ok(())
    }
}
