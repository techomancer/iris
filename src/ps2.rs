use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use winit::keyboard::KeyCode;
use std::io::Write;
use crate::traits::{Device, Resettable, Saveable};
use crate::snapshot::{get_field, toml_u8, toml_bool, hex_u8};
use crate::devlog::LogModule;

/// Callback trait for PS/2 interrupts
pub trait Ps2Callback: Send + Sync {
    /// Set the interrupt line state
    fn set_interrupt(&self, active: bool);
}

enum Ps2Source {
    Keyboard,
    Mouse,       // motion packet byte — counted in mouse_queue_bytes
    MouseCmd,    // command-response byte (ACK, BAT, ID) — not counted
}

#[derive(PartialEq)]
enum CommandState {
    Idle,
    SetLeds,
    SetScancodeSet,
    WriteConfig,
    SetTypematic,
    MouseData,  // consuming one data byte for a mouse command (e.g. F3 sample rate, E8 resolution)
}

struct Ps2State {
    rx_queue: VecDeque<(u8, Ps2Source)>,
    mouse_queue_bytes: usize,
    next_write_is_mouse: bool,
    led_state: u8,
    scancode_set: u8,
    config: u8,           // 8042 controller configuration byte
    command_state: CommandState,
    scanning_enabled: bool,
    mouse_enabled: bool,
    last_read: u8,
}

/// Combined PS/2 Keyboard and Mouse Controller
pub struct Ps2Controller {
    state: Mutex<Ps2State>,
    callback: Option<Arc<dyn Ps2Callback>>,
    running: AtomicBool,
}

impl Ps2Controller {
    /// Create a new PS/2 Controller
    pub fn new(callback: Option<Arc<dyn Ps2Callback>>) -> Self {
        Self {
            state: Mutex::new(Ps2State {
                rx_queue: VecDeque::new(),
                mouse_queue_bytes: 0,
                next_write_is_mouse: false,
                led_state: 0,
                scancode_set: 2, // Default to Set 2
                config: 0x47,    // Default: kbd IRQ + mouse IRQ + SYS flag
                command_state: CommandState::Idle,
                scanning_enabled: false,  // keyboard starts disabled; PROM enables via 0xF4
                mouse_enabled: false,     // mouse starts disabled; PROM enables via 0xF4
                last_read: 0xAA, // pretend we finished BAT at startup
            }),
            callback,
            running: AtomicBool::new(false),
        }
    }

    /// Update interrupt state based on queue status
    fn update_interrupt(&self) {
        if let Some(cb) = &self.callback {
            let state = self.state.lock();
            let len = state.rx_queue.len();
            drop(state);
            cb.set_interrupt(len > 0);
        }
    }

    /// Read a byte from the data port (0x40)
    pub fn read_data(&self) -> u8 {
        let mut state = self.state.lock();
        let val = if let Some((byte, src)) = state.rx_queue.pop_front() {
            if matches!(src, Ps2Source::Mouse) { state.mouse_queue_bytes -= 1; }
            state.last_read = byte;
            byte
        } else {
            state.last_read
        };
        drop(state);
        if crate::devlog::devlog_is_active(LogModule::Ps2) {
            dlog!(LogModule::Ps2, "PS2: Read Data -> {:02x}", val);
        }
        self.update_interrupt();
        val
    }

    /// Write a byte to the data port (0x40)
    pub fn write_data(&self, val: u8) {
        let mut state = self.state.lock();
        let dbg = crate::devlog::devlog_is_active(LogModule::Ps2);
        if state.next_write_is_mouse {
            if state.command_state == CommandState::MouseData {
                if dbg { dlog!(LogModule::Ps2, "PS2: Mouse data <- {:02x}", val); }
                state.rx_queue.push_back((0xFA, Ps2Source::MouseCmd));
                state.command_state = CommandState::Idle;
                state.next_write_is_mouse = false;
            } else {
                state.rx_queue.push_back((0xFA, Ps2Source::MouseCmd)); // ACK
                match val {
                    0xFF => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Mouse Reset <- {:02x}", val); }
                        state.mouse_enabled = false;
                        state.rx_queue.push_back((0xAA, Ps2Source::MouseCmd)); // BAT Success
                        state.rx_queue.push_back((0x00, Ps2Source::MouseCmd)); // ID
                        state.next_write_is_mouse = false;
                    }
                    0xF6 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Mouse Set Defaults <- {:02x}", val); }
                        state.mouse_enabled = false;
                        state.next_write_is_mouse = false;
                    }
                    0xF5 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Mouse Disable Data Reporting <- {:02x}", val); }
                        state.mouse_enabled = false;
                        state.next_write_is_mouse = false;
                    }
                    0xF4 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Mouse Enable Data Reporting <- {:02x}", val); }
                        state.mouse_enabled = true;
                        state.next_write_is_mouse = false;
                    }
                    0xF2 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Mouse Read ID <- {:02x}", val); }
                        state.rx_queue.push_back((0x00, Ps2Source::MouseCmd));
                        state.next_write_is_mouse = false;
                    }
                    0xF3 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Mouse Set Sample Rate <- {:02x}", val); }
                        state.command_state = CommandState::MouseData;
                    }
                    0xE8 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Mouse Set Resolution <- {:02x}", val); }
                        state.command_state = CommandState::MouseData;
                    }
                    _ => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Mouse unsupported <- {:02x}", val); }
                        state.next_write_is_mouse = false;
                    }
                }
            }
        } else {
            match state.command_state {
                CommandState::Idle => match val {
                    0xFF => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Reset <- {:02x}", val); }
                        state.rx_queue.clear(); state.mouse_queue_bytes = 0;
                        state.led_state = 0;
                        state.scancode_set = 2;
                        state.scanning_enabled = false; // reset leaves scanning disabled; PROM enables via F4
                        state.command_state = CommandState::Idle;
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard)); // ACK
                        state.rx_queue.push_back((0xAA, Ps2Source::Keyboard)); // BAT Success
                    }
                    0xED => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Set LEDs <- {:02x}", val); }
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                        state.command_state = CommandState::SetLeds;
                    }
                    0xF0 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Set/Get Scancode Set <- {:02x}", val); }
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                        state.command_state = CommandState::SetScancodeSet;
                    }
                    0xF4 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Enable Scanning <- {:02x}", val); }
                        state.scanning_enabled = true;
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                    }
                    0xF5 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Disable Scanning <- {:02x}", val); }
                        state.scanning_enabled = false;
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                    }
                    0x14 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Set Typematic Rate/Delay <- {:02x}", val); }
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                        state.command_state = CommandState::SetTypematic;
                    }
                    0xEE => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Echo <- {:02x}", val); }
                        state.rx_queue.push_back((0xEE, Ps2Source::Keyboard));
                    }
                    0xF2 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Read ID <- {:02x}", val); }
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                        state.rx_queue.push_back((0xAB, Ps2Source::Keyboard));
                        state.rx_queue.push_back((0x83, Ps2Source::Keyboard));
                    }
                    0xF6 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Reset to Defaults <- {:02x}", val); }
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                    }
                    0xFC => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Reset and Disable <- {:02x}", val); }
                        state.scanning_enabled = false;
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                    }
                    0x76 => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Reset to Defaults <- {:02x}", val); }
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                    }
                    0xFA => {
                        // ACK echoed back by PROM — ignore silently
                    }
                    _ => {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard unsupported <- {:02x}", val); }
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard)); // Default ACK
                    }
                },
                CommandState::SetLeds => {
                    if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard LEDs <- {:02x}", val); }
                    state.led_state = val;
                    state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                    state.command_state = CommandState::Idle;
                }
                CommandState::SetScancodeSet => {
                    if val == 0 {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Get Scancode Set <- {:02x}", val); }
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                        let current_set = state.scancode_set;
                        state.rx_queue.push_back((current_set, Ps2Source::Keyboard));
                    } else {
                        if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Set Scancode Set <- {:02x}", val); }
                        state.scancode_set = val;
                        state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                    }
                    state.command_state = CommandState::Idle;
                }
                CommandState::WriteConfig => {
                    if dbg { dlog!(LogModule::Ps2, "PS2: Config <- {:02x}", val); }
                    state.config = val;
                    state.command_state = CommandState::Idle;
                }
                CommandState::SetTypematic => {
                    if dbg { dlog!(LogModule::Ps2, "PS2: Keyboard Typematic <- {:02x}", val); }
                    state.rx_queue.push_back((0xFA, Ps2Source::Keyboard));
                    state.command_state = CommandState::Idle;
                }
                CommandState::MouseData => {
                    if dbg { dlog!(LogModule::Ps2, "PS2: Mouse data <- {:02x}", val); }
                    state.rx_queue.push_back((0xFA, Ps2Source::Mouse));
                    state.command_state = CommandState::Idle;
                }
            }
        }
        drop(state);
        self.update_interrupt();
    }

    /// Read from the status port (0x44)
    pub fn read_status(&self) -> u8 {
        let state = self.state.lock();
        let mut status = 0;
        if let Some((_, source)) = state.rx_queue.front() {
            status |= 0x01; // OBF (Output Buffer Full)
            if let Ps2Source::Mouse = source {
                status |= 0x20; // AUX (Mouse Data)
            }
        }
        status |= 0x04; // SYS (System Flag)
        if crate::devlog::devlog_is_active(LogModule::Ps2) {
            dlog!(LogModule::Ps2, "PS2: Read Status -> {:02x}", status);
        }
        status
    }

    /// Write to the command port (0x44)
    pub fn write_command(&self, val: u8) {
        let mut state = self.state.lock();
        match val {
            0xAA => {
                if crate::devlog::devlog_is_active(LogModule::Ps2) {
                    dlog!(LogModule::Ps2, "PS2: Command Self-Test (AA)");
                }
                state.rx_queue.push_back((0x55, Ps2Source::Keyboard)); // controller self-test pass
                state.rx_queue.push_back((0xAA, Ps2Source::Keyboard)); // keyboard BAT success
                state.next_write_is_mouse = false;
            }
            0x20 => {
                if crate::devlog::devlog_is_active(LogModule::Ps2) {
                    dlog!(LogModule::Ps2, "PS2: Command Read Config");
                }
                let cfg = state.config;
                state.rx_queue.push_back((cfg, Ps2Source::Keyboard));
            }
            0x60 => {
                if crate::devlog::devlog_is_active(LogModule::Ps2) {
                    dlog!(LogModule::Ps2, "PS2: Command Write Config");
                }
                state.command_state = CommandState::WriteConfig;
            }
            0xD4 => {
                if crate::devlog::devlog_is_active(LogModule::Ps2) {
                    dlog!(LogModule::Ps2, "PS2: Command Write Mouse");
                }
                state.next_write_is_mouse = true;
            }
            _ => {
                if crate::devlog::devlog_is_active(LogModule::Ps2) {
                    dlog!(LogModule::Ps2, "PS2: unsupported command {:02x}", val);
                }
                state.next_write_is_mouse = false;
            }
        }
        drop(state);
        self.update_interrupt();
    }

    /// Push a scancode byte to the keyboard queue (called from UI)
    pub fn push_kb(&self, key: KeyCode, pressed: bool) {
        if !self.running.load(Ordering::Relaxed) {
            eprintln!("PS2: push_kb({:?}, {}) DROPPED — not running", key, pressed);
            return;
        }
        let mut state = self.state.lock();
        if !state.scanning_enabled {
            eprintln!("PS2: push_kb({:?}, {}) DROPPED — scanning disabled", key, pressed);
            return;
        }
        eprintln!("PS2: push_kb({:?}, {}) — scanning set={}, queue_len={}", key, pressed, state.scancode_set, state.rx_queue.len());

        match state.scancode_set {
            1 => {
                if let Some(scancode) = self.map_keycode_set1(key) {
                    let val = scancode as u16;
                    if (val & 0xFF00) == 0xE000 {
                        state.rx_queue.push_back((0xE0, Ps2Source::Keyboard));
                        if crate::devlog::devlog_is_active(LogModule::Ps2) {
                            dlog!(LogModule::Ps2, "PS2: Pushed KB byte E0");
                        }
                    }
                    let mut code = (val & 0xFF) as u8;
                    if !pressed {
                        code |= 0x80;
                    }
                    if crate::devlog::devlog_is_active(LogModule::Ps2) {
                        dlog!(LogModule::Ps2, "PS2: Pushed KB byte {:02x}", code);
                    }
                    state.rx_queue.push_back((code, Ps2Source::Keyboard));
                }
            }
            2 => {
                if let Some(scancode) = self.map_keycode_set2(key) {
                    let val = scancode as u16;
                    if (val & 0xFF00) == 0xE000 {
                        state.rx_queue.push_back((0xE0, Ps2Source::Keyboard));
                        if crate::devlog::devlog_is_active(LogModule::Ps2) {
                            dlog!(LogModule::Ps2, "PS2: Pushed KB byte E0");
                        }
                    }
                    if !pressed {
                        state.rx_queue.push_back((0xF0, Ps2Source::Keyboard));
                        if crate::devlog::devlog_is_active(LogModule::Ps2) {
                            dlog!(LogModule::Ps2, "PS2: Pushed KB byte F0");
                        }
                    }
                    if crate::devlog::devlog_is_active(LogModule::Ps2) {
                        dlog!(LogModule::Ps2, "PS2: Pushed KB byte {:02x}", (val & 0xFF) as u8);
                    }
                    state.rx_queue.push_back(((val & 0xFF) as u8, Ps2Source::Keyboard));
                }
            }
            3 => {
                if let Some(scancode) = self.map_keycode_set3(key) {
                    let val = scancode as u8;
                    if !pressed {
                        state.rx_queue.push_back((0xF0, Ps2Source::Keyboard));
                        if crate::devlog::devlog_is_active(LogModule::Ps2) {
                            dlog!(LogModule::Ps2, "PS2: Pushed KB byte F0");
                        }
                    }
                    if crate::devlog::devlog_is_active(LogModule::Ps2) {
                        dlog!(LogModule::Ps2, "PS2: Pushed KB byte {:02x}", val);
                    }
                    state.rx_queue.push_back((val, Ps2Source::Keyboard));
                }
            }
            _ => {}
        }
        drop(state);
        self.update_interrupt();
    }

    fn map_keycode_set2(&self, code: KeyCode) -> Option<ScancodeSet2> {
        match code {
            KeyCode::KeyA => Some(ScancodeSet2::A),
            KeyCode::KeyB => Some(ScancodeSet2::B),
            KeyCode::KeyC => Some(ScancodeSet2::C),
            KeyCode::KeyD => Some(ScancodeSet2::D),
            KeyCode::KeyE => Some(ScancodeSet2::E),
            KeyCode::KeyF => Some(ScancodeSet2::F),
            KeyCode::KeyG => Some(ScancodeSet2::G),
            KeyCode::KeyH => Some(ScancodeSet2::H),
            KeyCode::KeyI => Some(ScancodeSet2::I),
            KeyCode::KeyJ => Some(ScancodeSet2::J),
            KeyCode::KeyK => Some(ScancodeSet2::K),
            KeyCode::KeyL => Some(ScancodeSet2::L),
            KeyCode::KeyM => Some(ScancodeSet2::M),
            KeyCode::KeyN => Some(ScancodeSet2::N),
            KeyCode::KeyO => Some(ScancodeSet2::O),
            KeyCode::KeyP => Some(ScancodeSet2::P),
            KeyCode::KeyQ => Some(ScancodeSet2::Q),
            KeyCode::KeyR => Some(ScancodeSet2::R),
            KeyCode::KeyS => Some(ScancodeSet2::S),
            KeyCode::KeyT => Some(ScancodeSet2::T),
            KeyCode::KeyU => Some(ScancodeSet2::U),
            KeyCode::KeyV => Some(ScancodeSet2::V),
            KeyCode::KeyW => Some(ScancodeSet2::W),
            KeyCode::KeyX => Some(ScancodeSet2::X),
            KeyCode::KeyY => Some(ScancodeSet2::Y),
            KeyCode::KeyZ => Some(ScancodeSet2::Z),
            KeyCode::Digit0 => Some(ScancodeSet2::Key0),
            KeyCode::Digit1 => Some(ScancodeSet2::Key1),
            KeyCode::Digit2 => Some(ScancodeSet2::Key2),
            KeyCode::Digit3 => Some(ScancodeSet2::Key3),
            KeyCode::Digit4 => Some(ScancodeSet2::Key4),
            KeyCode::Digit5 => Some(ScancodeSet2::Key5),
            KeyCode::Digit6 => Some(ScancodeSet2::Key6),
            KeyCode::Digit7 => Some(ScancodeSet2::Key7),
            KeyCode::Digit8 => Some(ScancodeSet2::Key8),
            KeyCode::Digit9 => Some(ScancodeSet2::Key9),
            KeyCode::Minus => Some(ScancodeSet2::Minus),
            KeyCode::Equal => Some(ScancodeSet2::Equals),
            KeyCode::BracketLeft => Some(ScancodeSet2::LBracket),
            KeyCode::BracketRight => Some(ScancodeSet2::RBracket),
            KeyCode::Backslash => Some(ScancodeSet2::Backslash),
            KeyCode::Semicolon => Some(ScancodeSet2::Semicolon),
            KeyCode::Quote => Some(ScancodeSet2::Quote),
            KeyCode::Comma => Some(ScancodeSet2::Comma),
            KeyCode::Period => Some(ScancodeSet2::Period),
            KeyCode::Slash => Some(ScancodeSet2::Slash),
            KeyCode::Backquote => Some(ScancodeSet2::Backquote),
            KeyCode::Space => Some(ScancodeSet2::Space),
            KeyCode::Enter => Some(ScancodeSet2::Enter),
            KeyCode::Backspace => Some(ScancodeSet2::Backspace),
            KeyCode::Escape => Some(ScancodeSet2::Esc),
            KeyCode::Tab => Some(ScancodeSet2::Tab),
            KeyCode::ShiftLeft => Some(ScancodeSet2::LShift),
            KeyCode::ShiftRight => Some(ScancodeSet2::RShift),
            KeyCode::ControlLeft => Some(ScancodeSet2::LCtrl),
            KeyCode::ControlRight => Some(ScancodeSet2::RCtrl),
            KeyCode::AltLeft => Some(ScancodeSet2::LAlt),
            KeyCode::AltRight => Some(ScancodeSet2::RAlt),
            KeyCode::ArrowUp => Some(ScancodeSet2::Up),
            KeyCode::ArrowDown => Some(ScancodeSet2::Down),
            KeyCode::ArrowLeft => Some(ScancodeSet2::Left),
            KeyCode::ArrowRight => Some(ScancodeSet2::Right),
            KeyCode::F1 => Some(ScancodeSet2::F1),
            KeyCode::F2 => Some(ScancodeSet2::F2),
            KeyCode::F3 => Some(ScancodeSet2::F3),
            KeyCode::F4 => Some(ScancodeSet2::F4),
            KeyCode::F5 => Some(ScancodeSet2::F5),
            KeyCode::F6 => Some(ScancodeSet2::F6),
            KeyCode::F7 => Some(ScancodeSet2::F7),
            KeyCode::F8 => Some(ScancodeSet2::F8),
            KeyCode::F9 => Some(ScancodeSet2::F9),
            KeyCode::F10 => Some(ScancodeSet2::F10),
            KeyCode::F11 => Some(ScancodeSet2::F11),
            KeyCode::F12 => Some(ScancodeSet2::F12),
            KeyCode::Numpad0 => Some(ScancodeSet2::Keypad0),
            KeyCode::Numpad1 => Some(ScancodeSet2::Keypad1),
            KeyCode::Numpad2 => Some(ScancodeSet2::Keypad2),
            KeyCode::Numpad3 => Some(ScancodeSet2::Keypad3),
            KeyCode::Numpad4 => Some(ScancodeSet2::Keypad4),
            KeyCode::Numpad5 => Some(ScancodeSet2::Keypad5),
            KeyCode::Numpad6 => Some(ScancodeSet2::Keypad6),
            KeyCode::Numpad7 => Some(ScancodeSet2::Keypad7),
            KeyCode::Numpad8 => Some(ScancodeSet2::Keypad8),
            KeyCode::Numpad9 => Some(ScancodeSet2::Keypad9),
            KeyCode::NumpadAdd => Some(ScancodeSet2::KeypadPlus),
            KeyCode::NumpadSubtract => Some(ScancodeSet2::KeypadMinus),
            KeyCode::NumpadMultiply => Some(ScancodeSet2::KeypadStar),
            KeyCode::NumpadDivide => Some(ScancodeSet2::KeypadSlash),
            KeyCode::NumpadDecimal => Some(ScancodeSet2::KeypadPeriod),
            KeyCode::NumpadEnter => Some(ScancodeSet2::KeypadEnter),
            KeyCode::NumLock => Some(ScancodeSet2::NumLock),
            KeyCode::ScrollLock => Some(ScancodeSet2::ScrollLock),
            KeyCode::SuperLeft => Some(ScancodeSet2::LGUI),
            KeyCode::SuperRight => Some(ScancodeSet2::RGUI),
            KeyCode::ContextMenu => Some(ScancodeSet2::Apps),
            KeyCode::Home => Some(ScancodeSet2::Home),
            KeyCode::End => Some(ScancodeSet2::End),
            KeyCode::PageUp => Some(ScancodeSet2::PageUp),
            KeyCode::PageDown => Some(ScancodeSet2::PageDown),
            KeyCode::Insert => Some(ScancodeSet2::Insert),
            KeyCode::Delete => Some(ScancodeSet2::Delete),
            _ => None,
        }
    }

    fn map_keycode_set1(&self, code: KeyCode) -> Option<ScancodeSet1> {
        match code {
            KeyCode::KeyA => Some(ScancodeSet1::A),
            KeyCode::KeyB => Some(ScancodeSet1::B),
            KeyCode::KeyC => Some(ScancodeSet1::C),
            KeyCode::KeyD => Some(ScancodeSet1::D),
            KeyCode::KeyE => Some(ScancodeSet1::E),
            KeyCode::KeyF => Some(ScancodeSet1::F),
            KeyCode::KeyG => Some(ScancodeSet1::G),
            KeyCode::KeyH => Some(ScancodeSet1::H),
            KeyCode::KeyI => Some(ScancodeSet1::I),
            KeyCode::KeyJ => Some(ScancodeSet1::J),
            KeyCode::KeyK => Some(ScancodeSet1::K),
            KeyCode::KeyL => Some(ScancodeSet1::L),
            KeyCode::KeyM => Some(ScancodeSet1::M),
            KeyCode::KeyN => Some(ScancodeSet1::N),
            KeyCode::KeyO => Some(ScancodeSet1::O),
            KeyCode::KeyP => Some(ScancodeSet1::P),
            KeyCode::KeyQ => Some(ScancodeSet1::Q),
            KeyCode::KeyR => Some(ScancodeSet1::R),
            KeyCode::KeyS => Some(ScancodeSet1::S),
            KeyCode::KeyT => Some(ScancodeSet1::T),
            KeyCode::KeyU => Some(ScancodeSet1::U),
            KeyCode::KeyV => Some(ScancodeSet1::V),
            KeyCode::KeyW => Some(ScancodeSet1::W),
            KeyCode::KeyX => Some(ScancodeSet1::X),
            KeyCode::KeyY => Some(ScancodeSet1::Y),
            KeyCode::KeyZ => Some(ScancodeSet1::Z),
            KeyCode::Digit0 => Some(ScancodeSet1::Key0),
            KeyCode::Digit1 => Some(ScancodeSet1::Key1),
            KeyCode::Digit2 => Some(ScancodeSet1::Key2),
            KeyCode::Digit3 => Some(ScancodeSet1::Key3),
            KeyCode::Digit4 => Some(ScancodeSet1::Key4),
            KeyCode::Digit5 => Some(ScancodeSet1::Key5),
            KeyCode::Digit6 => Some(ScancodeSet1::Key6),
            KeyCode::Digit7 => Some(ScancodeSet1::Key7),
            KeyCode::Digit8 => Some(ScancodeSet1::Key8),
            KeyCode::Digit9 => Some(ScancodeSet1::Key9),
            KeyCode::Minus => Some(ScancodeSet1::Minus),
            KeyCode::Equal => Some(ScancodeSet1::Equals),
            KeyCode::BracketLeft => Some(ScancodeSet1::LBracket),
            KeyCode::BracketRight => Some(ScancodeSet1::RBracket),
            KeyCode::Backslash => Some(ScancodeSet1::Backslash),
            KeyCode::Semicolon => Some(ScancodeSet1::Semicolon),
            KeyCode::Quote => Some(ScancodeSet1::Quote),
            KeyCode::Comma => Some(ScancodeSet1::Comma),
            KeyCode::Period => Some(ScancodeSet1::Period),
            KeyCode::Slash => Some(ScancodeSet1::Slash),
            KeyCode::Backquote => Some(ScancodeSet1::Backquote),
            KeyCode::Space => Some(ScancodeSet1::Space),
            KeyCode::Enter => Some(ScancodeSet1::Enter),
            KeyCode::Backspace => Some(ScancodeSet1::Backspace),
            KeyCode::Escape => Some(ScancodeSet1::Esc),
            KeyCode::Tab => Some(ScancodeSet1::Tab),
            KeyCode::ShiftLeft => Some(ScancodeSet1::LShift),
            KeyCode::ShiftRight => Some(ScancodeSet1::RShift),
            KeyCode::ControlLeft => Some(ScancodeSet1::LCtrl),
            KeyCode::ControlRight => Some(ScancodeSet1::RCtrl),
            KeyCode::AltLeft => Some(ScancodeSet1::LAlt),
            KeyCode::AltRight => Some(ScancodeSet1::RAlt),
            KeyCode::ArrowUp => Some(ScancodeSet1::Up),
            KeyCode::ArrowDown => Some(ScancodeSet1::Down),
            KeyCode::ArrowLeft => Some(ScancodeSet1::Left),
            KeyCode::ArrowRight => Some(ScancodeSet1::Right),
            KeyCode::F1 => Some(ScancodeSet1::F1),
            KeyCode::F2 => Some(ScancodeSet1::F2),
            KeyCode::F3 => Some(ScancodeSet1::F3),
            KeyCode::F4 => Some(ScancodeSet1::F4),
            KeyCode::F5 => Some(ScancodeSet1::F5),
            KeyCode::F6 => Some(ScancodeSet1::F6),
            KeyCode::F7 => Some(ScancodeSet1::F7),
            KeyCode::F8 => Some(ScancodeSet1::F8),
            KeyCode::F9 => Some(ScancodeSet1::F9),
            KeyCode::F10 => Some(ScancodeSet1::F10),
            KeyCode::F11 => Some(ScancodeSet1::F11),
            KeyCode::F12 => Some(ScancodeSet1::F12),
            KeyCode::Numpad0 => Some(ScancodeSet1::Keypad0),
            KeyCode::Numpad1 => Some(ScancodeSet1::Keypad1),
            KeyCode::Numpad2 => Some(ScancodeSet1::Keypad2),
            KeyCode::Numpad3 => Some(ScancodeSet1::Keypad3),
            KeyCode::Numpad4 => Some(ScancodeSet1::Keypad4),
            KeyCode::Numpad5 => Some(ScancodeSet1::Keypad5),
            KeyCode::Numpad6 => Some(ScancodeSet1::Keypad6),
            KeyCode::Numpad7 => Some(ScancodeSet1::Keypad7),
            KeyCode::Numpad8 => Some(ScancodeSet1::Keypad8),
            KeyCode::Numpad9 => Some(ScancodeSet1::Keypad9),
            KeyCode::NumpadAdd => Some(ScancodeSet1::KeypadPlus),
            KeyCode::NumpadSubtract => Some(ScancodeSet1::KeypadMinus),
            KeyCode::NumpadMultiply => Some(ScancodeSet1::KeypadStar),
            KeyCode::NumpadDivide => Some(ScancodeSet1::KeypadSlash),
            KeyCode::NumpadDecimal => Some(ScancodeSet1::KeypadPeriod),
            KeyCode::NumpadEnter => Some(ScancodeSet1::KeypadEnter),
            KeyCode::NumLock => Some(ScancodeSet1::NumLock),
            KeyCode::ScrollLock => Some(ScancodeSet1::ScrollLock),
            KeyCode::SuperLeft => Some(ScancodeSet1::LGUI),
            KeyCode::SuperRight => Some(ScancodeSet1::RGUI),
            KeyCode::ContextMenu => Some(ScancodeSet1::Apps),
            KeyCode::Home => Some(ScancodeSet1::Home),
            KeyCode::End => Some(ScancodeSet1::End),
            KeyCode::PageUp => Some(ScancodeSet1::PageUp),
            KeyCode::PageDown => Some(ScancodeSet1::PageDown),
            KeyCode::Insert => Some(ScancodeSet1::Insert),
            KeyCode::Delete => Some(ScancodeSet1::Delete),
            _ => None,
        }
    }

    fn map_keycode_set3(&self, code: KeyCode) -> Option<ScancodeSet3> {
        match code {
            KeyCode::KeyA => Some(ScancodeSet3::A),
            KeyCode::KeyB => Some(ScancodeSet3::B),
            KeyCode::KeyC => Some(ScancodeSet3::C),
            KeyCode::KeyD => Some(ScancodeSet3::D),
            KeyCode::KeyE => Some(ScancodeSet3::E),
            KeyCode::KeyF => Some(ScancodeSet3::F),
            KeyCode::KeyG => Some(ScancodeSet3::G),
            KeyCode::KeyH => Some(ScancodeSet3::H),
            KeyCode::KeyI => Some(ScancodeSet3::I),
            KeyCode::KeyJ => Some(ScancodeSet3::J),
            KeyCode::KeyK => Some(ScancodeSet3::K),
            KeyCode::KeyL => Some(ScancodeSet3::L),
            KeyCode::KeyM => Some(ScancodeSet3::M),
            KeyCode::KeyN => Some(ScancodeSet3::N),
            KeyCode::KeyO => Some(ScancodeSet3::O),
            KeyCode::KeyP => Some(ScancodeSet3::P),
            KeyCode::KeyQ => Some(ScancodeSet3::Q),
            KeyCode::KeyR => Some(ScancodeSet3::R),
            KeyCode::KeyS => Some(ScancodeSet3::S),
            KeyCode::KeyT => Some(ScancodeSet3::T),
            KeyCode::KeyU => Some(ScancodeSet3::U),
            KeyCode::KeyV => Some(ScancodeSet3::V),
            KeyCode::KeyW => Some(ScancodeSet3::W),
            KeyCode::KeyX => Some(ScancodeSet3::X),
            KeyCode::KeyY => Some(ScancodeSet3::Y),
            KeyCode::KeyZ => Some(ScancodeSet3::Z),
            KeyCode::Digit0 => Some(ScancodeSet3::Key0),
            KeyCode::Digit1 => Some(ScancodeSet3::Key1),
            KeyCode::Digit2 => Some(ScancodeSet3::Key2),
            KeyCode::Digit3 => Some(ScancodeSet3::Key3),
            KeyCode::Digit4 => Some(ScancodeSet3::Key4),
            KeyCode::Digit5 => Some(ScancodeSet3::Key5),
            KeyCode::Digit6 => Some(ScancodeSet3::Key6),
            KeyCode::Digit7 => Some(ScancodeSet3::Key7),
            KeyCode::Digit8 => Some(ScancodeSet3::Key8),
            KeyCode::Digit9 => Some(ScancodeSet3::Key9),
            KeyCode::Minus => Some(ScancodeSet3::Minus),
            KeyCode::Equal => Some(ScancodeSet3::Equals),
            KeyCode::BracketLeft => Some(ScancodeSet3::LBracket),
            KeyCode::BracketRight => Some(ScancodeSet3::RBracket),
            KeyCode::Backslash => Some(ScancodeSet3::Backslash),
            KeyCode::Semicolon => Some(ScancodeSet3::Semicolon),
            KeyCode::Quote => Some(ScancodeSet3::Quote),
            KeyCode::Comma => Some(ScancodeSet3::Comma),
            KeyCode::Period => Some(ScancodeSet3::Period),
            KeyCode::Slash => Some(ScancodeSet3::Slash),
            KeyCode::Backquote => Some(ScancodeSet3::Backquote),
            KeyCode::Space => Some(ScancodeSet3::Space),
            KeyCode::Enter => Some(ScancodeSet3::Enter),
            KeyCode::Backspace => Some(ScancodeSet3::Backspace),
            KeyCode::Escape => Some(ScancodeSet3::Esc),
            KeyCode::Tab => Some(ScancodeSet3::Tab),
            KeyCode::ShiftLeft => Some(ScancodeSet3::LShift),
            KeyCode::ShiftRight => Some(ScancodeSet3::RShift),
            KeyCode::ControlLeft => Some(ScancodeSet3::LCtrl),
            KeyCode::ControlRight => Some(ScancodeSet3::RCtrl),
            KeyCode::AltLeft => Some(ScancodeSet3::LAlt),
            KeyCode::AltRight => Some(ScancodeSet3::RAlt),
            KeyCode::ArrowUp => Some(ScancodeSet3::Up),
            KeyCode::ArrowDown => Some(ScancodeSet3::Down),
            KeyCode::ArrowLeft => Some(ScancodeSet3::Left),
            KeyCode::ArrowRight => Some(ScancodeSet3::Right),
            KeyCode::F1 => Some(ScancodeSet3::F1),
            KeyCode::F2 => Some(ScancodeSet3::F2),
            KeyCode::F3 => Some(ScancodeSet3::F3),
            KeyCode::F4 => Some(ScancodeSet3::F4),
            KeyCode::F5 => Some(ScancodeSet3::F5),
            KeyCode::F6 => Some(ScancodeSet3::F6),
            KeyCode::F7 => Some(ScancodeSet3::F7),
            KeyCode::F8 => Some(ScancodeSet3::F8),
            KeyCode::F9 => Some(ScancodeSet3::F9),
            KeyCode::F10 => Some(ScancodeSet3::F10),
            KeyCode::F11 => Some(ScancodeSet3::F11),
            KeyCode::F12 => Some(ScancodeSet3::F12),
            KeyCode::Numpad0 => Some(ScancodeSet3::Keypad0),
            KeyCode::Numpad1 => Some(ScancodeSet3::Keypad1),
            KeyCode::Numpad2 => Some(ScancodeSet3::Keypad2),
            KeyCode::Numpad3 => Some(ScancodeSet3::Keypad3),
            KeyCode::Numpad4 => Some(ScancodeSet3::Keypad4),
            KeyCode::Numpad5 => Some(ScancodeSet3::Keypad5),
            KeyCode::Numpad6 => Some(ScancodeSet3::Keypad6),
            KeyCode::Numpad7 => Some(ScancodeSet3::Keypad7),
            KeyCode::Numpad8 => Some(ScancodeSet3::Keypad8),
            KeyCode::Numpad9 => Some(ScancodeSet3::Keypad9),
            KeyCode::NumpadAdd => Some(ScancodeSet3::KeypadPlus),
            KeyCode::NumpadSubtract => Some(ScancodeSet3::KeypadMinus),
            KeyCode::NumpadMultiply => Some(ScancodeSet3::KeypadStar),
            KeyCode::NumpadDivide => Some(ScancodeSet3::KeypadSlash),
            KeyCode::NumpadDecimal => Some(ScancodeSet3::KeypadPeriod),
            KeyCode::NumpadEnter => Some(ScancodeSet3::KeypadEnter),
            KeyCode::NumLock => Some(ScancodeSet3::NumLock),
            KeyCode::ScrollLock => Some(ScancodeSet3::ScrollLock),
            KeyCode::SuperLeft => Some(ScancodeSet3::LGUI),
            KeyCode::SuperRight => Some(ScancodeSet3::RGUI),
            KeyCode::ContextMenu => Some(ScancodeSet3::Apps),
            KeyCode::Home => Some(ScancodeSet3::Home),
            KeyCode::End => Some(ScancodeSet3::End),
            KeyCode::PageUp => Some(ScancodeSet3::PageUp),
            KeyCode::PageDown => Some(ScancodeSet3::PageDown),
            KeyCode::Insert => Some(ScancodeSet3::Insert),
            KeyCode::Delete => Some(ScancodeSet3::Delete),
            _ => None,
        }
    }

    /// Push a byte to the mouse queue (called from UI)
    pub fn push_mouse(&self, byte: u8) {
        if !self.running.load(Ordering::Relaxed) { return; }
        let mut state = self.state.lock();
        if !state.mouse_enabled {
            return;
        }
        state.rx_queue.push_back((byte, Ps2Source::Mouse));
        state.mouse_queue_bytes += 1;
        drop(state);
        if crate::devlog::devlog_is_active(LogModule::Ps2) {
            dlog!(LogModule::Ps2, "PS2: Pushed Mouse byte {:02x}", byte);
        }
        self.update_interrupt();
    }

    /// Push a 3-byte mouse motion packet, coalescing into the tail when the
    /// queue is congested (≥8 pending mouse packets) and the tail has the
    /// same button state.
    pub fn push_mouse_packet(&self, b0: u8, b1: u8, b2: u8) {
        if !self.running.load(Ordering::Relaxed) {
            eprintln!("PS2: mouse packet DROPPED — not running");
            return;
        }
        let mut state = self.state.lock();
        if !state.mouse_enabled {
            eprintln!("PS2: mouse packet DROPPED — mouse disabled");
            return;
        }
        eprintln!("PS2: mouse packet b0={:02x} b1={:02x} b2={:02x} queue_len={}", b0, b1, b2, state.rx_queue.len());
        let mouse_packets = state.mouse_queue_bytes / 3;
        let coalesced = if mouse_packets >= 8 {
            let len = state.rx_queue.len();
            let i = len.wrapping_sub(3);
            len >= 3
                && matches!(state.rx_queue[i],     (_, Ps2Source::Mouse))
                && matches!(state.rx_queue[i + 1], (_, Ps2Source::Mouse))
                && matches!(state.rx_queue[i + 2], (_, Ps2Source::Mouse))
                && (state.rx_queue[i].0 & 0x07) == (b0 & 0x07)
                && {
                    // Reconstruct 9-bit signed values from sign bit + data byte.
                    let old_dx = if state.rx_queue[i].0 & 0x10 != 0 { state.rx_queue[i + 1].0 as i32 - 256 } else { state.rx_queue[i + 1].0 as i32 };
                    let old_dy = if state.rx_queue[i].0 & 0x20 != 0 { state.rx_queue[i + 2].0 as i32 - 256 } else { state.rx_queue[i + 2].0 as i32 };
                    let new_b1_signed = if b0 & 0x10 != 0 { b1 as i32 - 256 } else { b1 as i32 };
                    let new_b2_signed = if b0 & 0x20 != 0 { b2 as i32 - 256 } else { b2 as i32 };
                    let raw_dx = old_dx + new_b1_signed;
                    let raw_dy = old_dy + new_b2_signed;
                    let new_dx = raw_dx.clamp(-256, 255);
                    let new_dy = raw_dy.clamp(-256, 255);
                    let mut new_b0 = 0x08u8 | (b0 & 0x07);
                    if new_dx < 0 { new_b0 |= 0x10; }
                    if new_dy < 0 { new_b0 |= 0x20; }
                    if raw_dx != new_dx { new_b0 |= 0x40; }
                    if raw_dy != new_dy { new_b0 |= 0x80; }
                    state.rx_queue[i].0     = new_b0;
                    state.rx_queue[i + 1].0 = new_dx as u8;
                    state.rx_queue[i + 2].0 = new_dy as u8;
                    true
                }
        } else {
            false
        };
        if !coalesced {
            state.rx_queue.push_back((b0, Ps2Source::Mouse));
            state.rx_queue.push_back((b1, Ps2Source::Mouse));
            state.rx_queue.push_back((b2, Ps2Source::Mouse));
            state.mouse_queue_bytes += 3;
        }
        drop(state);
        self.update_interrupt();
    }

    pub fn is_state_locked(&self) -> bool {
        self.state.is_locked()
    }
}

impl Device for Ps2Controller {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) { self.running.store(false, Ordering::Relaxed); }
    fn start(&self) { self.running.store(true, Ordering::Relaxed); }
    fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("ps2".to_string(), "PS/2 commands: ps2 debug <on|off> [DEV]".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd == "ps2" {
            if !args.is_empty() && args[0] == "debug" {
                let val = match args.get(1).map(|s| *s) {
                    Some("on") => true,
                    Some("off") => false,
                    _ => return Err("Usage: ps2 debug <on|off>".to_string()),
                };
                if val { crate::devlog::devlog().enable(LogModule::Ps2); } else { crate::devlog::devlog().disable(LogModule::Ps2); }
                writeln!(writer, "PS/2 debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                return Ok(());
            }
            return Err("Usage: ps2 debug <on|off>".to_string());
        }
        Err("Command not found".to_string())
    }
}

impl Resettable for Ps2Controller {
    fn power_on(&self) {
        self.running.store(true, Ordering::Relaxed);
        let mut state = self.state.lock();
        state.rx_queue.clear(); state.mouse_queue_bytes = 0;
        state.next_write_is_mouse = false;
        state.led_state = 0;
        state.scancode_set = 2;
        state.config = 0x47;
        state.command_state = CommandState::Idle;
        state.scanning_enabled = false;
        state.mouse_enabled = false;
        state.last_read = 0xAA;
    }
}

impl Saveable for Ps2Controller {
    fn save_state(&self) -> toml::Value {
        let state = self.state.lock();
        let mut tbl = toml::map::Map::new();
        
        let queue: Vec<toml::Value> = state.rx_queue.iter().map(|(byte, src)| {
            let src_val = match src { Ps2Source::Keyboard => 0, Ps2Source::Mouse => 1, Ps2Source::MouseCmd => 2 };
            let mut entry = toml::map::Map::new();
            entry.insert("byte".into(), hex_u8(*byte));
            entry.insert("src".into(), toml::Value::Integer(src_val));
            toml::Value::Table(entry)
        }).collect();
        tbl.insert("rx_queue".into(), toml::Value::Array(queue));

        tbl.insert("next_write_is_mouse".into(), toml::Value::Boolean(state.next_write_is_mouse));
        tbl.insert("led_state".into(), hex_u8(state.led_state));
        tbl.insert("scancode_set".into(), hex_u8(state.scancode_set));
        tbl.insert("config".into(), hex_u8(state.config));
        
        let cmd_state = match state.command_state {
            CommandState::Idle => 0,
            CommandState::SetLeds => 1,
            CommandState::SetScancodeSet => 2,
            CommandState::WriteConfig => 3,
            CommandState::SetTypematic => 4,
            CommandState::MouseData => 5,
        };
        tbl.insert("command_state".into(), toml::Value::Integer(cmd_state));
        
        tbl.insert("scanning_enabled".into(), toml::Value::Boolean(state.scanning_enabled));
        tbl.insert("mouse_enabled".into(), toml::Value::Boolean(state.mouse_enabled));
        tbl.insert("last_read".into(), hex_u8(state.last_read));

        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut state = self.state.lock();
        
        if let Some(q) = get_field(v, "rx_queue") {
            if let toml::Value::Array(arr) = q {
                state.rx_queue.clear(); state.mouse_queue_bytes = 0;
                for item in arr {
                    if let (Some(b), Some(s)) = (get_field(item, "byte"), get_field(item, "src")) {
                        let byte = toml_u8(b).unwrap_or(0);
                        let src = match s.as_integer().unwrap_or(0) {
                            1 => Ps2Source::Mouse,
                            2 => Ps2Source::MouseCmd,
                            _ => Ps2Source::Keyboard,
                        };
                        if matches!(src, Ps2Source::Mouse) { state.mouse_queue_bytes += 1; }
                        state.rx_queue.push_back((byte, src));
                    }
                }
            }
        }

        if let Some(x) = get_field(v, "next_write_is_mouse") { state.next_write_is_mouse = toml_bool(x).unwrap_or(false); }
        if let Some(x) = get_field(v, "led_state") { state.led_state = toml_u8(x).unwrap_or(0); }
        if let Some(x) = get_field(v, "scancode_set") { state.scancode_set = toml_u8(x).unwrap_or(2); }
        if let Some(x) = get_field(v, "config") { state.config = toml_u8(x).unwrap_or(0x47); }
        
        if let Some(x) = get_field(v, "command_state") {
            state.command_state = match x.as_integer().unwrap_or(0) {
                1 => CommandState::SetLeds,
                2 => CommandState::SetScancodeSet,
                3 => CommandState::WriteConfig,
                4 => CommandState::SetTypematic,
                5 => CommandState::MouseData,
                _ => CommandState::Idle,
            };
        }

        if let Some(x) = get_field(v, "scanning_enabled") { state.scanning_enabled = toml_bool(x).unwrap_or(false); }
        if let Some(x) = get_field(v, "mouse_enabled") { state.mouse_enabled = toml_bool(x).unwrap_or(false); }
        if let Some(x) = get_field(v, "last_read") { state.last_read = toml_u8(x).unwrap_or(0xAA); }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ScancodeSet1 {
    Esc = 0x01,
    Key1 = 0x02,
    Key2 = 0x03,
    Key3 = 0x04,
    Key4 = 0x05,
    Key5 = 0x06,
    Key6 = 0x07,
    Key7 = 0x08,
    Key8 = 0x09,
    Key9 = 0x0A,
    Key0 = 0x0B,
    Minus = 0x0C,
    Equals = 0x0D,
    Backspace = 0x0E,
    Tab = 0x0F,
    Q = 0x10,
    W = 0x11,
    E = 0x12,
    R = 0x13,
    T = 0x14,
    Y = 0x15,
    U = 0x16,
    I = 0x17,
    O = 0x18,
    P = 0x19,
    LBracket = 0x1A,
    RBracket = 0x1B,
    Enter = 0x1C,
    LCtrl = 0x1D,
    A = 0x1E,
    S = 0x1F,
    D = 0x20,
    F = 0x21,
    G = 0x22,
    H = 0x23,
    J = 0x24,
    K = 0x25,
    L = 0x26,
    Semicolon = 0x27,
    Quote = 0x28,
    Backquote = 0x29,
    LShift = 0x2A,
    Backslash = 0x2B,
    Z = 0x2C,
    X = 0x2D,
    C = 0x2E,
    V = 0x2F,
    B = 0x30,
    N = 0x31,
    M = 0x32,
    Comma = 0x33,
    Period = 0x34,
    Slash = 0x35,
    RShift = 0x36,
    KeypadStar = 0x37,
    LAlt = 0x38,
    Space = 0x39,
    CapsLock = 0x3A,
    F1 = 0x3B,
    F2 = 0x3C,
    F3 = 0x3D,
    F4 = 0x3E,
    F5 = 0x3F,
    F6 = 0x40,
    F7 = 0x41,
    F8 = 0x42,
    F9 = 0x43,
    F10 = 0x44,
    NumLock = 0x45,
    ScrollLock = 0x46,
    Keypad7 = 0x47,
    Keypad8 = 0x48,
    Keypad9 = 0x49,
    KeypadMinus = 0x4A,
    Keypad4 = 0x4B,
    Keypad5 = 0x4C,
    Keypad6 = 0x4D,
    KeypadPlus = 0x4E,
    Keypad1 = 0x4F,
    Keypad2 = 0x50,
    Keypad3 = 0x51,
    Keypad0 = 0x52,
    KeypadPeriod = 0x53,
    F11 = 0x57,
    F12 = 0x58,

    // Extended
    KeypadEnter = 0xE01C,
    RCtrl = 0xE01D,
    KeypadSlash = 0xE035,
    RAlt = 0xE038,
    Home = 0xE047,
    Up = 0xE048,
    PageUp = 0xE049,
    Left = 0xE04B,
    Right = 0xE04D,
    End = 0xE04F,
    Down = 0xE050,
    PageDown = 0xE051,
    Insert = 0xE052,
    Delete = 0xE053,
    LGUI = 0xE05B,
    RGUI = 0xE05C,
    Apps = 0xE05D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ScancodeSet2 {
    F9 = 0x01,
    F5 = 0x03,
    F3 = 0x04,
    F1 = 0x05,
    F2 = 0x06,
    F12 = 0x07,
    F10 = 0x09,
    F8 = 0x0A,
    F6 = 0x0B,
    F4 = 0x0C,
    Tab = 0x0D,
    Backquote = 0x0E,
    LAlt = 0x11,
    LShift = 0x12,
    LCtrl = 0x14,
    Q = 0x15,
    Key1 = 0x16,
    Z = 0x1A,
    S = 0x1B,
    A = 0x1C,
    W = 0x1D,
    Key2 = 0x1E,
    C = 0x21,
    X = 0x22,
    D = 0x23,
    E = 0x24,
    Key4 = 0x25,
    Key3 = 0x26,
    Space = 0x29,
    V = 0x2A,
    F = 0x2B,
    T = 0x2C,
    R = 0x2D,
    Key5 = 0x2E,
    N = 0x31,
    B = 0x32,
    H = 0x33,
    G = 0x34,
    Y = 0x35,
    Key6 = 0x36,
    M = 0x3A,
    J = 0x3B,
    U = 0x3C,
    Key7 = 0x3D,
    Key8 = 0x3E,
    Comma = 0x41,
    K = 0x42,
    I = 0x43,
    O = 0x44,
    Key0 = 0x45,
    Key9 = 0x46,
    Period = 0x49,
    Slash = 0x4A,
    L = 0x4B,
    Semicolon = 0x4C,
    P = 0x4D,
    Minus = 0x4E,
    Quote = 0x52,
    LBracket = 0x54,
    Equals = 0x55,
    CapsLock = 0x58,
    RShift = 0x59,
    Enter = 0x5A,
    RBracket = 0x5B,
    Backslash = 0x5D,
    Backspace = 0x66,
    Keypad1 = 0x69,
    Keypad4 = 0x6B,
    Keypad7 = 0x6C,
    KeypadPeriod = 0x71,
    Keypad0 = 0x70,
    Keypad2 = 0x72,
    Keypad5 = 0x73,
    Keypad6 = 0x74,
    Keypad8 = 0x75,
    Esc = 0x76,
    NumLock = 0x77,
    F11 = 0x78,
    KeypadPlus = 0x79,
    Keypad3 = 0x7A,
    KeypadMinus = 0x7B,
    KeypadStar = 0x7C,
    Keypad9 = 0x7D,
    ScrollLock = 0x7E,
    F7 = 0x83,

    // Extended
    End = 0xE069,
    Left = 0xE06B,
    Home = 0xE06C,
    Insert = 0xE070,
    Delete = 0xE071,
    Down = 0xE072,
    Right = 0xE074,
    Up = 0xE075,
    PageDown = 0xE07A,
    PageUp = 0xE07D,
    RAlt = 0xE011,
    RCtrl = 0xE014,
    KeypadSlash = 0xE04A,
    KeypadEnter = 0xE05A,
    LGUI = 0xE01F,
    RGUI = 0xE027,
    Apps = 0xE02F,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScancodeSet3 {
    A = 0x1C,
    B = 0x32,
    C = 0x21,
    D = 0x23,
    E = 0x24,
    F = 0x2B,
    G = 0x34,
    H = 0x33,
    I = 0x43,
    J = 0x3B,
    K = 0x42,
    L = 0x4B,
    M = 0x3A,
    N = 0x31,
    O = 0x44,
    P = 0x4D,
    Q = 0x15,
    R = 0x2D,
    S = 0x1B,
    T = 0x2C,
    U = 0x3C,
    V = 0x2A,
    W = 0x1D,
    X = 0x22,
    Y = 0x35,
    Z = 0x1A,
    Key0 = 0x45,
    Key1 = 0x16,
    Key2 = 0x1E,
    Key3 = 0x26,
    Key4 = 0x25,
    Key5 = 0x2E,
    Key6 = 0x36,
    Key7 = 0x3D,
    Key8 = 0x3E,
    Key9 = 0x46,
    Backquote = 0x0E,
    Minus = 0x4E,
    Equals = 0x55,
    Backslash = 0x5C,
    Backspace = 0x66,
    Space = 0x29,
    Tab = 0x0D,
    CapsLock = 0x14,
    LShift = 0x12,
    LCtrl = 0x11,
    LAlt = 0x19,
    RShift = 0x59,
    RCtrl = 0x58,
    RAlt = 0x39,
    Enter = 0x5A,
    Esc = 0x08,
    F1 = 0x07,
    F2 = 0x0F,
    F3 = 0x17,
    F4 = 0x1F,
    F5 = 0x27,
    F6 = 0x2F,
    F7 = 0x37,
    F8 = 0x3F,
    F9 = 0x47,
    F10 = 0x4F,
    F11 = 0x56,
    F12 = 0x5E,
    PrintScreen = 0x57,
    ScrollLock = 0x5F,
    Pause = 0x62,
    Insert = 0x67,
    Home = 0x6E,
    PageUp = 0x6F,
    Delete = 0x64,
    End = 0x65,
    PageDown = 0x6D,
    Up = 0x63,
    Left = 0x61,
    Down = 0x60,
    Right = 0x6A,
    NumLock = 0x76,
    KeypadSlash = 0x77,
    KeypadStar = 0x7E,
    KeypadMinus = 0x84,
    KeypadPlus = 0x7C,
    KeypadEnter = 0x79,
    KeypadPeriod = 0x71,
    Keypad0 = 0x70,
    Keypad1 = 0x69,
    Keypad2 = 0x72,
    Keypad3 = 0x7A,
    Keypad4 = 0x6B,
    Keypad5 = 0x73,
    Keypad6 = 0x74,
    Keypad7 = 0x6C,
    Keypad8 = 0x75,
    Keypad9 = 0x7D,
    LBracket = 0x54,
    RBracket = 0x5B,
    Semicolon = 0x4C,
    Quote = 0x52,
    Comma = 0x41,
    Period = 0x49,
    Slash = 0x4A,
    LGUI = 0x8B,
    RGUI = 0x8C,
    Apps = 0x8D,
}
