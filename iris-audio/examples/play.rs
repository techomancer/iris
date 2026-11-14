//! Play a steady tone until told to stop, and say what the stream is doing
//! meanwhile: for watching a backend follow the host while the default
//! output device is switched, unplugged or disabled.
//!
//! ```text
//! cargo run --release --example play -- [options]
//!   --backend default|pulse|wasapi|coreaudio|null|file:PATH   where to play (default)
//!   --rate HZ          the guest's rate (44100)
//!   --freq HZ          the tone (440)
//!   --amp LEVEL        tone level 0..1 (0.1; 0 plays silence)
//!   --prebuffer MS     OutputConfig::prebuffer (10)
//!   --latency MS       OutputConfig::device_latency (backend's default)
//!   --secs N           stop by itself after this long (never)
//!   --fallback on|off  OutputConfig::null_fallback: start with no device (off)
//! ```
//!
//! Press Enter to stop. A line is printed whenever the device or its format
//! changes; the line below it is the stream's state, refreshed four times a
//! second. A steady tone makes any gap or click easy to hear. Unlike
//! `tone`, the guest here sleeps between writes instead of spinning, so it
//! can be left running.

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::{Duration, Instant};

use iris_audio::{Audio, BackendChoice, OutputConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = BackendChoice::Default;
    let mut cfg = OutputConfig { name: "continuous tone".into(), rate: 44100, ..Default::default() };
    let (mut freq, mut amp, mut secs) = (440.0f64, 0.1f64, f64::INFINITY);

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--backend" => {
                backend = match value.as_str() {
                    "default" => BackendChoice::Default,
                    "pulse" => BackendChoice::Pulse,
                    "wasapi" => BackendChoice::Wasapi,
                    "coreaudio" => BackendChoice::CoreAudio,
                    "null" => BackendChoice::Null,
                    v => BackendChoice::File(v.strip_prefix("file:").ok_or("--backend default|pulse|wasapi|coreaudio|null|file:PATH")?.into()),
                }
            }
            "--rate" => cfg.rate = value.parse()?,
            "--freq" => freq = value.parse()?,
            "--amp" => amp = value.parse()?,
            "--prebuffer" => cfg.prebuffer = Duration::from_millis(value.parse()?),
            "--latency" => cfg.device_latency = Some(Duration::from_millis(value.parse()?)),
            "--fallback" => cfg.null_fallback = value == "on",
            "--secs" => secs = value.parse()?,
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }

    let audio = Audio::new(backend)?;
    let rate = f64::from(cfg.rate);
    let mut out = audio.open_output(cfg)?;
    println!("{}: playing {freq} Hz at level {amp}; press Enter to stop", audio.backend());

    // Enter, or the end of input, stops the tone.
    let stop = Arc::new(AtomicBool::new(false));
    let stop_t = stop.clone();
    std::thread::spawn(move || {
        let _ = std::io::stdin().read_line(&mut String::new());
        stop_t.store(true, Relaxed);
    });

    let start = Instant::now();
    let (mut written, mut phase) = (0u64, 0.0f64);
    let mut block = Vec::new();
    let mut shown = None;
    let mut report = Duration::ZERO;
    while !stop.load(Relaxed) && start.elapsed().as_secs_f64() < secs {
        // The guest: every millisecond or so, the frames that are due.
        let now = start.elapsed();
        let due = (now.as_secs_f64() * rate) as u64;
        block.clear();
        for _ in written..due {
            phase = (phase + 2.0 * std::f64::consts::PI * freq / rate) % (2.0 * std::f64::consts::PI);
            let v = (phase.sin() * amp) as f32;
            block.extend([v, v]);
        }
        out.write(&block);
        written = due;

        let s = out.stats();
        let device = (out.device().unwrap_or_default(), s.device);
        if shown.as_ref() != Some(&device) {
            let name = if device.0.is_empty() { "no device" } else { &device.0 };
            let format = device.1.map_or(String::new(), |f| format!(", {} Hz, {} channels", f.rate, f.channels));
            println!("\r{:<110}", format!("{:7.1}s  {name}{format}", now.as_secs_f64()));
            shown = Some(device);
        }
        if now >= report {
            report = now + Duration::from_millis(250);
            print!(
                "\r{:7.1}s  queued {:5.1} ms  device {}  adjust {:+.3}%  underruns {}  overruns {}  dropped {}  device underruns {}   ",
                now.as_secs_f64(),
                s.queued.as_secs_f64() * 1e3,
                s.device_delay.map_or("    ? ms".into(), |d| format!("{:5.1} ms", d.as_secs_f64() * 1e3)),
                s.adjust * 100.0,
                s.underruns,
                s.overruns,
                s.dropped,
                s.device_underruns,
            );
            let _ = std::io::stdout().flush();
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    out.idle();
    std::thread::sleep(Duration::from_millis(200));
    let s = out.stats();
    println!("\n{} frames written, {} played", s.written, s.played);
    Ok(())
}
