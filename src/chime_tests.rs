//! Each sound card's boot chime, from the frames the card itself plays.
//! A chime is reduced to one `level:note` entry per 100 ms of sound and
//! compared with a table recorded from our own run of each machine.

use crate::config::{AudioHost, GraphicsBoard, MachineConfig, MachineProfile};
use crate::dev::audio_dump::tests::{dump, TAP};
use std::time::{Duration, Instant};

/// Boot the embedded PROM with no disk until its chime has played: sound
/// heard, then `quiet` frames of silence or nothing new for five seconds.
/// Returns the card's rate and its non-silent frames (the O2's gaps vary with load).
fn boot(profile: MachineProfile, cpu: &str, quiet: usize, edit: fn(&mut MachineConfig)) -> (u32, Vec<(i16, i16)>) {
    let cpu = cpu.to_string();
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            let dir = crate::test_temp::TempPath::dir("chime");
            let (tap, bytes) = dump();
            TAP.with(|t| *t.borrow_mut() = Some(tap));
            let mut cfg = crate::bench_runner::bench_config([64, 0, 0, 0], &crate::cpu::parts::R4400_150);
            cfg.machine.profile = profile;
            cfg.machine.cpu = Some(cpu);
            cfg.no_audio = false;
            cfg.audio.host = AudioHost::Null;
            crate::test_temp::nvram_in(&mut cfg, &dir);
            let held = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
            cfg.monitor_port = Some(held.local_addr().unwrap().port());
            crate::dev::serial_host::tests::free_ports(&mut cfg);
            edit(&mut cfg);
            let mut m = crate::workstation::build(cfg).expect("machine builds");
            m.start();
            m.cpu_start();
            let deadline = Instant::now() + Duration::from_secs(1700);
            let (mut seen, mut still) = (0, 0);
            let played = loop {
                std::thread::sleep(Duration::from_millis(250));
                let recs = bytes.records();
                let all: Vec<_> = recs.iter().flat_map(|r| r.1.iter().copied()).collect();
                still = if all.len() == seen { still + 1 } else { 0 };
                seen = all.len();
                let tail = all.iter().rev().take_while(|&&f| f == (0, 0)).count();
                if all.iter().any(|&f| f != (0, 0)) && (tail >= quiet || still >= 20) {
                    let rate = recs[0].0;
                    assert!(recs.iter().all(|r| r.0 == rate), "the card changed rate");
                    break (rate, all.into_iter().filter(|&f| f != (0, 0)).collect());
                }
                assert!(Instant::now() < deadline, "the chime never finished");
            };
            m.stop();
            played
        })
        .unwrap()
        .join()
        .unwrap()
}

/// Per 100 ms of sound: the level (log2 of the RMS, 6 dB steps) and the
/// loudest semitone (MIDI number, by Goertzel over a Hann window).
fn table(rate: u32, frames: &[(i16, i16)]) -> String {
    let n = rate as usize / 10;
    frames
        .chunks_exact(n)
        .map(|w| {
            let x: Vec<f64> = w.iter().enumerate().map(|(i, &(l, r))| {
                (l as f64 + r as f64) / 2.0 * (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos())
            }).collect();
            let rms = (w.iter().map(|&(l, r)| ((l as f64 + r as f64) / 2.0).powi(2)).sum::<f64>() / n as f64).sqrt();
            let power = |note: u32| {
                let f = 440.0 * 2f64.powf((note as f64 - 69.0) / 12.0);
                let c = 2.0 * (std::f64::consts::TAU * f / rate as f64).cos();
                let (mut a, mut b) = (0.0, 0.0);
                for v in &x {
                    (a, b) = (v + c * a - b, a);
                }
                a * a + b * b - c * a * b
            };
            let note = (24..=108).filter(|&k| 440.0 * 2f64.powf((k as f64 - 69.0) / 12.0) < rate as f64 / 2.0)
                .max_by(|&p, &q| power(p).total_cmp(&power(q))).unwrap();
            format!("{}:{note}", (rms as u32).max(1).ilog2())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Windows may differ by a step in level or a semitone, and a fifth of them by more:
/// partials of near equal strength trade places as the chime decays.
fn check(rate: u32, want_rate: u32, frames: &[(i16, i16)], want: &str) {
    let got = table(rate, frames);
    let entries = |t: &str| -> Vec<(u32, u32)> {
        t.split_whitespace().map(|e| { let (l, n) = e.split_once(':').unwrap(); (l.parse().unwrap(), n.parse().unwrap()) }).collect()
    };
    let (g, w) = (entries(&got), entries(want));
    let off = g.iter().zip(&w).filter(|(g, w)| g.0.abs_diff(w.0) > 1 || g.1.abs_diff(w.1) > 1).count();
    assert!(rate == want_rate && g.len().abs_diff(w.len()) <= 1 && off * 5 <= w.len(), "{rate} Hz, {} windows, {off} off:\n{got}", g.len());
}

#[test]
fn indy_hal2_chime() {
    let (rate, frames) = boot(MachineProfile::IndyIp24, "r4400", 11025, |_| {});
    check(rate, 44100, &frames, "13:35 10:59 13:38 12:55 12:49 11:76 13:35 11:57 11:88 9:88 9:57 8:88 11:47 12:54 12:55 13:33 12:76 12:91 12:98 10:98");
}

#[test]
fn o2_mace_chime() {
    let (rate, frames) = boot(MachineProfile::O2Ip32, "r5000", 34000, |_| {});
    check(rate, 34000, &frames, "9:100 9:104 9:101 9:60 10:104 10:62 9:62 9:62 8:62 8:62 7:62 7:62 6:62 6:62 6:62 6:62 5:62 5:63");
}

// With the graphics board off the PROM stops at a keypress prompt.
#[test]
fn octane_rad1_chime() {
    let (rate, frames) = boot(MachineProfile::OctaneIp30, "r10000", 4000, |c| c.graphics.board = Some(GraphicsBoard::None));
    check(rate, 32000, &frames, "12:37 12:52 12:52 12:52 12:76 13:76 12:52 12:52 12:52 12:60 12:60 12:55 12:55 12:55 12:55 11:60 10:60 8:60 6:60");
}

#[test]
fn indigo2_ip22_hal2_chime() {
    let (rate, frames) = boot(MachineProfile::Indigo2Ip22, "r4400", 11025, |_| {});
    check(rate, 44100, &frames, "11:95 11:61 11:61 11:61 12:69 12:69 12:69 11:69 11:69 11:85 11:76 11:69 11:69 11:69 11:69 11:69 11:69 11:61 10:61");
}

#[test]
fn indigo2_ip28_hal2_chime() {
    let (rate, frames) = boot(MachineProfile::Indigo2Ip28, "r10000", 11025, |_| {});
    check(rate, 44100, &frames, "11:76 12:73 12:73 12:76 12:88 12:73 12:73 11:76 12:76 12:76 12:73 12:73 11:73 11:76 11:76 11:76 11:73 11:73 11:73");
}
