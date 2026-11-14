//! Play a test tone the way an emulated sound device would, and report
//! how the stream held up.
//!
//! ```text
//! cargo run --release --example tone -- [options]
//!   --backend default|pulse|wasapi|coreaudio|null|file:PATH   where to play (default)
//!   --rate HZ          the guest's rate (44100)
//!   --burst FRAMES     frames the guest writes at a time (4; try 1024)
//!   --drift PPM        how far the guest's clock is off (0; try 2000)
//!   --jitter MS        the guest's thread stalls this long now and then (0)
//!   --prebuffer MS     OutputConfig::prebuffer (10)
//!   --latency MS       OutputConfig::device_latency (backend's default)
//!   --amp LEVEL        tone level 0..1 (0.1; 0 plays silence)
//!   --secs N           how long (5)
//!   --tee PATH         also record what the device was given
//!   --fallback on|off  OutputConfig::null_fallback: start with no device (off)
//! ```
//!
//! `--burst 4` is a codec fed from a timer interrupt; `--burst 1024` one
//! fed from a DMA buffer. The guest's thread spins between writes, as the
//! emulator's timer thread does, so its pace is as even as the host allows.

use std::time::{Duration, Instant};

use iris_audio::{Audio, BackendChoice, OutputConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = BackendChoice::Default;
    let mut cfg = OutputConfig { name: "test tone".into(), rate: 44100, ..Default::default() };
    let (mut burst, mut drift, mut jitter, mut amp, mut secs) = (4usize, 0.0f64, 0u64, 0.1f64, 5.0f64);

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
            "--burst" => burst = value.parse()?,
            "--drift" => drift = value.parse()?,
            "--jitter" => jitter = value.parse()?,
            "--prebuffer" => cfg.prebuffer = Duration::from_millis(value.parse()?),
            "--latency" => cfg.device_latency = Some(Duration::from_millis(value.parse()?)),
            "--amp" => amp = value.parse()?,
            "--fallback" => cfg.null_fallback = value == "on",
            "--secs" => secs = value.parse()?,
            "--tee" => cfg.tee = Some(value.into()),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }

    let audio = Audio::new(backend)?;
    let rate = cfg.rate;
    let mut out = audio.open_output(cfg)?;
    println!("{}: {}", audio.backend(), out.device().unwrap_or_default());

    let guest_hz = f64::from(rate) * (1.0 + drift / 1e6);
    let start = Instant::now();
    let (mut written, mut phase) = (0u64, 0.0f64);
    let mut block = vec![0i16; burst * 2];
    let mut report = Duration::from_secs(1);
    let mut stall = Duration::from_millis(700);
    while start.elapsed().as_secs_f64() < secs {
        let now = start.elapsed();
        if (written + burst as u64) as f64 <= now.as_secs_f64() * guest_hz {
            for f in block.chunks_mut(2) {
                phase += 2.0 * std::f64::consts::PI * 440.0 / f64::from(rate);
                let v = (phase.sin() * amp * 32767.0) as i16;
                f.copy_from_slice(&[v, v]);
            }
            out.write(&block);
            written += burst as u64;
        } else {
            std::hint::spin_loop();
        }
        if jitter > 0 && now >= stall {
            std::thread::sleep(Duration::from_millis(jitter));
            stall += Duration::from_millis(700);
        }
        if now >= report {
            report += Duration::from_secs(1);
            let s = out.stats();
            println!(
                "{:>3}s  queued {:5.1} ms  device {}  adjust {:+.3}%  underruns {}  overruns {}  dropped {}  device underruns {}",
                now.as_secs(),
                s.queued.as_secs_f64() * 1e3,
                s.device_delay.map_or("    ? ms".into(), |d| format!("{:5.1} ms", d.as_secs_f64() * 1e3)),
                s.adjust * 100.0,
                s.underruns,
                s.overruns,
                s.dropped,
                s.device_underruns,
            );
        }
    }
    out.idle();
    std::thread::sleep(Duration::from_millis(200));
    let s = out.stats();
    println!("device format {:?}; {} frames written, {} played", s.device, s.written, s.played);
    Ok(())
}
