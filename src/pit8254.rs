use std::sync::Arc;
use spin::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, Device, Resettable, Saveable};
use crate::snapshot::{get_field, toml_u16, toml_u32, toml_u8, toml_bool, hex_u16, hex_u32, hex_u8};
use crate::hptimer::{TimerManager, TimerId, TimerReturn};
use std::io::Write;

pub trait TimerCallback: Send + Sync {
    fn callback(&self);
}

struct Channel {
    // Registers
    count: u16, // Current count value (simulated)
    reload: u16, // Reload value
    latched_count: Option<u16>,

    // Configuration
    mode: u8,
    rw_mode: u8, // 0: Latch, 1: LSB, 2: MSB, 3: LSB+MSB
    bcd: bool,

    // Internal State
    rw_state: u8, // 0: First byte, 1: Second byte (for rw_mode 3)

    // Timer tracking (for count interpolation between callbacks)
    period_start: Option<Instant>,
    period_duration: Option<Duration>,
    input_freq: u32,
    // Active hptimer ID, if armed
    timer_id: Option<TimerId>,
}

impl Channel {
    fn new() -> Self {
        Self {
            count: 0,
            reload: 0,
            latched_count: None,
            mode: 0,
            rw_mode: 1, // Default LSB
            bcd: false,
            rw_state: 0,
            period_start: None,
            period_duration: None,
            input_freq: 0,
            timer_id: None,
        }
    }
}

#[derive(Clone)]
pub struct Pit8254 {
    channels: [Arc<Mutex<Channel>>; 3],
    callbacks: [Option<Arc<dyn TimerCallback>>; 3],
    timer_manager: Arc<std::sync::OnceLock<Arc<TimerManager>>>,
    debug: Arc<AtomicBool>,
    base_frequency: u32,
}

impl Pit8254 {
    pub fn new(base_frequency: u32, cb0: Option<Arc<dyn TimerCallback>>, cb1: Option<Arc<dyn TimerCallback>>, cb2: Option<Arc<dyn TimerCallback>>) -> Self {
        let pit = Self {
            channels: [
                Arc::new(Mutex::new(Channel::new())),
                Arc::new(Mutex::new(Channel::new())),
                Arc::new(Mutex::new(Channel::new())),
            ],
            callbacks: [cb0, cb1, cb2],
            timer_manager: Arc::new(std::sync::OnceLock::new()),
            debug: Arc::new(AtomicBool::new(false)),
            base_frequency,
        };
        // Channel 2 is driven by the base frequency (1 MHz)
        pit.channels[2].lock().input_freq = base_frequency;
        pit
    }

    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timer_manager.set(tm);
    }

    fn arm_channel(&self, idx: usize) {
        let Some(tm) = self.timer_manager.get() else { return; };

        // Disarm any existing timer first
        self.disarm_channel(idx);

        let period;
        {
            let mut chan = self.channels[idx].lock();
            if chan.reload == 0 || chan.input_freq == 0 {
                return;
            }
            let ns = 1_000_000_000u64 / chan.input_freq as u64;
            period = Duration::from_nanos(chan.reload as u64 * ns);
            chan.period_start = Some(Instant::now());
            chan.period_duration = Some(period);
        }

        let chan_arc = self.channels[idx].clone();
        let callback = self.callbacks[idx].clone();
        let debug = self.debug.clone();

        let id = tm.add_recurring(Instant::now() + period, period, (), move |_| {
            {
                let mut chan = chan_arc.lock();
                chan.count = 0;
                chan.period_start = Some(Instant::now());
            }
            if let Some(cb) = &callback {
                if debug.load(Ordering::Relaxed) && idx != 2 {
                    println!("PIT: Channel {} expired, triggering callback", idx);
                }
                cb.callback();
            } else if debug.load(Ordering::Relaxed) && idx != 2 {
                println!("PIT: Channel {} expired (no callback)", idx);
            }
            TimerReturn::Continue
        });

        self.channels[idx].lock().timer_id = Some(id);
    }

    fn disarm_channel(&self, idx: usize) {
        let id = self.channels[idx].lock().timer_id.take();
        if let Some(id) = id {
            if let Some(tm) = self.timer_manager.get() {
                tm.remove(id);
            }
        }
        let mut chan = self.channels[idx].lock();
        chan.period_start = None;
        chan.period_duration = None;
    }

    fn read_channel(&self, idx: usize) -> u8 {
        let mut chan = self.channels[idx].lock();

        // If latched, read from latched value
        let val = if let Some(latched) = chan.latched_count {
            latched
        } else {
            // Calculate current count based on elapsed time
            if let (Some(start), Some(duration)) = (chan.period_start, chan.period_duration) {
                if chan.input_freq > 0 {
                    let elapsed = start.elapsed();
                    if elapsed < duration {
                        let remaining = duration - elapsed;
                        let ns_per_tick = 1_000_000_000 / chan.input_freq as u128;
                        let ticks = (remaining.as_nanos() / ns_per_tick) as u16;
                        ticks
                    } else {
                        0
                    }
                } else {
                    chan.count
                }
            } else {
                chan.count
            }
        };

        match chan.rw_mode {
            1 => (val & 0xFF) as u8, // LSB
            2 => (val >> 8) as u8,   // MSB
            3 => { // LSB then MSB
                if chan.rw_state == 0 {
                    chan.rw_state = 1;
                    (val & 0xFF) as u8
                } else {
                    chan.rw_state = 0;
                    chan.latched_count = None; // Clear latch after full read
                    (val >> 8) as u8
                }
            }
            _ => 0,
        }
    }

    fn write_channel(&self, idx: usize, val: u8) {
        let mut update_chain = false;
        let mut do_arm = false;
        {
            let mut chan = self.channels[idx].lock();

            match chan.rw_mode {
                1 => { // LSB
                    chan.reload = (chan.reload & 0xFF00) | (val as u16);
                    chan.count = chan.reload;
                    do_arm = true;
                }
                2 => { // MSB
                    chan.reload = (chan.reload & 0x00FF) | ((val as u16) << 8);
                    chan.count = chan.reload;
                    do_arm = true;
                }
                3 => { // LSB then MSB
                    if chan.rw_state == 0 {
                        chan.reload = (chan.reload & 0xFF00) | (val as u16);
                        chan.rw_state = 1;
                    } else {
                        chan.reload = (chan.reload & 0x00FF) | ((val as u16) << 8);
                        chan.rw_state = 0;
                        chan.count = chan.reload;
                        do_arm = true;
                    }
                }
                _ => {}
            }
            if idx == 2 {
                update_chain = true;
            }
        } // release lock before arm_channel / update_chaining

        if do_arm && !update_chain {
            self.arm_channel(idx);
        }
        if update_chain {
            self.update_chaining();
        }
    }

    fn write_control(&self, val: u8) {
        let sc = (val >> 6) & 0x3;
        let rw = (val >> 4) & 0x3;
        let m = (val >> 1) & 0x7;
        let bcd = (val & 1) != 0;

        if sc == 3 {
            // Read-Back Command (ignored for now)
            return;
        }

        let idx = sc as usize;
        let mut update_chain = false;
        {
            let mut chan = self.channels[idx].lock();

            if rw == 0 {
                // Counter Latch Command
                if chan.latched_count.is_none() {
                    // Calculate current count
                    let current = if let (Some(start), Some(duration)) = (chan.period_start, chan.period_duration) {
                        let elapsed = start.elapsed();
                        if elapsed < duration {
                            let remaining = duration - elapsed;
                            let ns_per_tick = 1_000_000_000 / self.base_frequency as u128;
                            (remaining.as_nanos() / ns_per_tick) as u16
                        } else {
                            0
                        }
                    } else {
                        chan.count
                    };
                    chan.latched_count = Some(current);
                }
            } else {
                // Mode/RW setup
                chan.rw_mode = rw;
                chan.mode = m;
                chan.bcd = bcd;
                chan.rw_state = 0;
                chan.latched_count = None;
                if idx == 2 {
                    update_chain = true;
                }
            }
        }

        if update_chain {
            self.update_chaining();
        }
    }

    fn update_chaining(&self) {
        // Channel 2 drives Channel 0 and 1.
        // If Ch2 is in Mode 2 (Rate Generator) or Mode 3 (Square Wave), it generates a clock.
        // Frequency = Input Freq / Reload Value.
        let freq = {
            let chan = self.channels[2].lock();
            let mode_periodic = chan.mode == 2 || chan.mode == 3;
            if mode_periodic && chan.reload > 0 && chan.input_freq > 0 {
                chan.input_freq / chan.reload as u32
            } else {
                0
            }
        };

        // Re-arm channel 2 first (its own config may have changed)
        self.arm_channel(2);

        for i in 0..2 {
            let changed = {
                let mut chan = self.channels[i].lock();
                if chan.input_freq != freq {
                    if self.debug.load(Ordering::Relaxed) {
                        println!("PIT: Ch{} input freq updated to {} Hz (driven by Ch2)", i, freq);
                    }
                    chan.input_freq = freq;
                    true
                } else {
                    false
                }
            };
            if changed {
                self.arm_channel(i);
            }
        }
    }

    pub fn read(&self, addr: u32) -> BusRead8 {
        let val = match addr {
            0 => self.read_channel(0),
            1 => self.read_channel(1),
            2 => self.read_channel(2),
            _ => return BusRead8::err(),
        };
        if self.debug.load(Ordering::Relaxed) {
            println!("PIT: Read addr {} -> {:02x}", addr, val);
        }
        BusRead8::ok(val)
    }

    pub fn write(&self, addr: u32, val: u8) -> u32 {
        if self.debug.load(Ordering::Relaxed) {
            println!("PIT: Write addr {} val {:02x}", addr, val);
        }
        match addr {
            0 => self.write_channel(0, val),
            1 => self.write_channel(1, val),
            2 => self.write_channel(2, val),
            3 => self.write_control(val),
            _ => return BUS_ERR,
        }
        BUS_OK
    }
}

impl Device for Pit8254 {
    fn step(&self, _cycles: u64) {}

    fn stop(&self) {
        for i in 0..3 {
            self.disarm_channel(i);
        }
    }

    fn start(&self) {

        for i in 0..3 {
            self.arm_channel(i);
        }
    }

    fn is_running(&self) -> bool { self.timer_manager.get().is_some() }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("pit".to_string(), "PIT commands: pit status | pit debug <on|off> [DEV]".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd == "pit" {
            if args.is_empty() {
                return Err("Usage: pit <debug|status> ...".to_string());
            }
            match args[0] {
                "debug" => {
                    let val = match args.get(1).map(|s| *s) {
                        Some("on") => true,
                        Some("off") => false,
                        _ => return Err("Usage: pit debug <on|off>".to_string()),
                    };
                    self.debug.store(val, Ordering::Relaxed);
                    writeln!(writer, "PIT debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                    Ok(())
                }
                "status" => {
                    writeln!(writer, "PIT Status:").unwrap();
                    for (i, channel_arc) in self.channels.iter().enumerate() {
                        let chan = channel_arc.lock();
                        writeln!(writer, "  Channel {}: Mode={} RW={} BCD={} Count={:04x} Reload={:04x} Freq={}Hz Running={}",
                            i, chan.mode, chan.rw_mode, chan.bcd, chan.count, chan.reload, chan.input_freq, chan.period_start.is_some()).unwrap();
                    }
                    Ok(())
                }
                _ => Err("Usage: pit <debug|status> ...".to_string()),
            }
        } else {
            Err("Command not found".to_string())
        }
    }
}

// ============================================================================
// Resettable + Saveable for Pit8254
// ============================================================================

impl Resettable for Pit8254 {
    /// Reset all channel registers to power-on defaults.
    fn power_on(&self) {
        // Disarm all timers before resetting state
        for i in 0..3 {
            self.disarm_channel(i);
        }
        for (i, channel_arc) in self.channels.iter().enumerate() {
            let mut chan = channel_arc.lock();
            chan.count = 0;
            chan.reload = 0;
            chan.latched_count = None;
            chan.mode = 0;
            chan.rw_mode = 1; // default LSB
            chan.bcd = false;
            chan.rw_state = 0;
            chan.period_start = None;
            chan.period_duration = None;
            chan.timer_id = None;
            // Channel 2 is driven by base frequency; channels 0,1 start with 0 until chaining fires.
            chan.input_freq = if i == 2 { self.base_frequency } else { 0 };
        }
    }
}

fn chan_to_toml(chan: &Channel) -> toml::Value {
    let mut t = toml::map::Map::new();
    t.insert("count".into(),      hex_u16(chan.count));
    t.insert("reload".into(),     hex_u16(chan.reload));
    t.insert("mode".into(),       hex_u8(chan.mode));
    t.insert("rw_mode".into(),    hex_u8(chan.rw_mode));
    t.insert("bcd".into(),        toml::Value::Boolean(chan.bcd));
    t.insert("input_freq".into(), hex_u32(chan.input_freq));
    toml::Value::Table(t)
}

fn chan_from_toml(v: &toml::Value, chan: &mut Channel) {
    if let Some(x) = get_field(v, "count")      { if let Some(n) = toml_u16(x) { chan.count = n; } }
    if let Some(x) = get_field(v, "reload")     { if let Some(n) = toml_u16(x) { chan.reload = n; } }
    if let Some(x) = get_field(v, "mode")       { if let Some(n) = toml_u8(x)  { chan.mode = n; } }
    if let Some(x) = get_field(v, "rw_mode")    { if let Some(n) = toml_u8(x)  { chan.rw_mode = n; } }
    if let Some(x) = get_field(v, "bcd")        { if let Some(b) = toml_bool(x) { chan.bcd = b; } }
    if let Some(x) = get_field(v, "input_freq") { if let Some(n) = toml_u32(x) { chan.input_freq = n; } }
    // Transient state cleared on load.
    chan.latched_count = None;
    chan.rw_state = 0;
    chan.period_start = None;
    chan.period_duration = None;
    chan.timer_id = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as AOrdering};
    use std::sync::Mutex;
    use std::thread;
    use std::time::{Duration, Instant};

    // Serialise all timing-sensitive tests so they don't interfere with each other
    // when the test suite runs with multiple threads.
    static SERIAL: Mutex<()> = Mutex::new(());

    struct CountCallback(Arc<AtomicU32>);
    impl TimerCallback for CountCallback {
        fn callback(&self) { self.0.fetch_add(1, AOrdering::SeqCst); }
    }

    fn make_pit(freq: u32) -> Pit8254 {
        let pit = Pit8254::new(freq, None, None, None);
        pit.set_timer_manager(Arc::new(TimerManager::new()));
        pit
    }

    fn make_pit_cb(freq: u32, cb: Arc<dyn TimerCallback>) -> Pit8254 {
        let pit = Pit8254::new(freq, None, None, Some(cb));
        pit.set_timer_manager(Arc::new(TimerManager::new()));
        pit
    }

    // Program channel `ch` as mode 2 (rate generator), LSB+MSB, with given reload.
    fn program_mode2(pit: &Pit8254, ch: u8, reload: u16) {
        // Control word: SC=ch, RW=3 (LSB+MSB), M=2, BCD=0
        let ctrl = (ch << 6) | (3 << 4) | (2 << 1);
        pit.write(3, ctrl);
        pit.write(ch as u32, (reload & 0xFF) as u8);
        pit.write(ch as u32, (reload >> 8) as u8);
    }

    // Latch and read a 16-bit count from channel `ch`.
    fn latch_read16(pit: &Pit8254, ch: u8) -> u16 {
        // Latch command: SC=ch, RW=0
        let ctrl = (ch << 6) | 0;
        pit.write(3, ctrl);
        let lo = { let _r = pit.read(ch as u32); if _r.is_ok() { let v = _r.data; v } else { panic!("bad read") } };
        let hi = { let _r = pit.read(ch as u32); if _r.is_ok() { let v = _r.data; v } else { panic!("bad read") } };
        (hi as u16) << 8 | lo as u16
    }

    // Read channel without latching (LSB+MSB mode 3).
    fn read16(pit: &Pit8254, ch: u8) -> u16 {
        let lo = { let _r = pit.read(ch as u32); if _r.is_ok() { let v = _r.data; v } else { panic!("bad read") } };
        let hi = { let _r = pit.read(ch as u32); if _r.is_ok() { let v = _r.data; v } else { panic!("bad read") } };
        (hi as u16) << 8 | lo as u16
    }

    // Start PIT, program channel 2, then wait briefly for the timer to be armed.
    fn start_and_program(freq: u32, reload: u16) -> Pit8254 {
        let pit = make_pit(freq);
        pit.start();
        program_mode2(&pit, 2, reload);
        // Give the timer manager a moment to process the new timer.
        thread::sleep(Duration::from_millis(2));
        pit
    }

    fn start_and_program_cb(freq: u32, reload: u16, cb: Arc<dyn TimerCallback>) -> Pit8254 {
        let pit = make_pit_cb(freq, cb);
        pit.start();
        program_mode2(&pit, 2, reload);
        thread::sleep(Duration::from_millis(2));
        pit
    }

    /// Channel 2 at 1 MHz, reload=100: one period = 100 µs wall-clock.
    /// After waiting well past one period the count must be < reload (timer running).
    #[test]
    fn test_ch2_100us_period() {
        let _lock = SERIAL.lock().unwrap();
        let pit = start_and_program(1_000_000, 100);
        // Sleep several extra periods so we're definitely mid-period.
        thread::sleep(Duration::from_millis(1));
        let count = latch_read16(&pit, 2);
        pit.stop();
        // In mode 2 the channel reloads; count is always in [0, reload).
        assert!(count < 100, "count={} should be < reload=100", count);
    }

    /// Mid-period read: after ~5 ms into a 10 ms period count should be ~5000.
    #[test]
    fn test_ch2_midperiod_read() {
        let _lock = SERIAL.lock().unwrap();
        let pit = start_and_program(1_000_000, 10_000); // 10 ms period
        thread::sleep(Duration::from_millis(5));
        let count = latch_read16(&pit, 2);
        pit.stop();
        // startup sleep (~2ms) + 5ms measured = ~7ms elapsed, remaining ~3000. Allow ±1000.
        let expected: i32 = 3000;
        let delta = (count as i32 - expected).abs();
        assert!(delta < 1000, "count={} expected ~{} (delta={})", count, expected, delta);
    }

    /// Callback fires ~100 times per 100 ms when reload=1000 at 1 MHz (1 ms period).
    #[test]
    fn test_ch2_callback_rate() {
        let _lock = SERIAL.lock().unwrap();
        let counter = Arc::new(AtomicU32::new(0));
        let cb = Arc::new(CountCallback(counter.clone())) as Arc<dyn TimerCallback>;
        let pit = start_and_program_cb(1_000_000, 1000, cb);
        thread::sleep(Duration::from_millis(100));
        pit.stop();
        let fires = counter.load(AOrdering::SeqCst);
        // ~100 callbacks in 100 ms, allow ±25 for jitter.
        assert!(fires >= 75 && fires <= 125,
            "callback fired {} times in 100 ms, expected ~100", fires);
    }

    /// The count decrements monotonically within a period.
    #[test]
    fn test_ch2_count_decrements() {
        let _lock = SERIAL.lock().unwrap();
        let pit = start_and_program(1_000_000, 0xFFFF); // ~65 ms period
        let c0 = latch_read16(&pit, 2);
        thread::sleep(Duration::from_millis(5));
        let c1 = latch_read16(&pit, 2);
        pit.stop();
        // c0 > c1 (count decrements toward 0)
        assert!(c0 > c1, "count should decrement: c0={} c1={}", c0, c1);
    }

    /// Reload=0xFFFF at 1 MHz: period = 65.535 ms. After 10 ms count ≈ 55535.
    #[test]
    fn test_ch2_large_reload() {
        let _lock = SERIAL.lock().unwrap();
        let pit = start_and_program(1_000_000, 0xFFFF);
        thread::sleep(Duration::from_millis(10));
        let count = latch_read16(&pit, 2);
        pit.stop();
        // startup sleep (~2ms) + 10ms measured = ~12ms total elapsed, remaining ~53535
        let expected: i32 = 0xFFFF - 12_000;
        let delta = (count as i32 - expected).abs();
        assert!(delta < 1000,
            "count={} expected ~{} (delta={})", count, expected, delta);
    }
}

impl Saveable for Pit8254 {
    fn save_state(&self) -> toml::Value {
        let mut tbl = toml::map::Map::new();
        for (i, channel_arc) in self.channels.iter().enumerate() {
            let chan = channel_arc.lock();
            tbl.insert(format!("ch{}", i), chan_to_toml(&chan));
        }
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        for (i, channel_arc) in self.channels.iter().enumerate() {
            let mut chan = channel_arc.lock();
            if let Some(ct) = get_field(v, &format!("ch{}", i)) {
                chan_from_toml(ct, &mut chan);
            }
        }
        Ok(())
    }
}
