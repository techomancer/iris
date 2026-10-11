use super::proto::decode::*;
use super::proto::{self, packet, track_event as te};
use super::*;

/// Sessions and the enable mask are process-global.
static SERIAL: Mutex<()> = Mutex::new(());

static T_SLICE: EventDef = EventDef::new("test slice", cat::MISC).with_arg("n");
static T_INSTANT: EventDef = EventDef::new("test instant", cat::IRQ).on_track("test irq");
static T_COUNTER: EventDef = EventDef::new("test counter", cat::GFX);
static T_GFX: EventDef = EventDef::new("test gfx", cat::GFX);

fn temp_path(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!("iris-perfetto-{tag}-{}.pftrace", std::process::id()))
        .to_string_lossy()
        .into_owned()
}

/// Top-level `TracePacket`s of a trace file.
fn packets(data: &[u8]) -> Vec<Vec<(u32, Value<'_>)>> {
    fields(data)
        .into_iter()
        .map(|(f, v)| {
            assert_eq!(f, proto::TRACE_PACKET);
            match v { Value::Bytes(b) => fields(b), other => panic!("packet is {other:?}") }
        })
        .collect()
}

struct Decoded {
    /// (timestamp, type, track uuid, name, counter value, arg)
    events: Vec<(u64, u64, u64, Option<String>, Option<i64>, Option<u64>)>,
    /// uuid -> (name, is_thread, is_counter)
    tracks: std::collections::HashMap<u64, (String, bool, bool)>,
}

fn decode(data: &[u8]) -> Decoded {
    let pk = packets(data);
    assert!(!pk.is_empty());
    let flags0 = get_u64(&pk[0], packet::SEQUENCE_FLAGS).unwrap();
    assert_ne!(flags0 & packet::SEQ_INCREMENTAL_STATE_CLEARED, 0, "first packet must clear state");

    let mut names = std::collections::HashMap::new();
    let mut d = Decoded { events: Vec::new(), tracks: Default::default() };
    for p in &pk {
        assert_eq!(get_u64(p, packet::TRUSTED_PACKET_SEQUENCE_ID), Some(1));
        if let Some(id) = get_msg(p, packet::INTERNED_DATA) {
            for (f, v) in &id {
                if *f == proto::interned_data::EVENT_NAMES {
                    let Value::Bytes(b) = v else { panic!() };
                    let m = fields(b);
                    let iid = get_u64(&m, 1).unwrap();
                    assert!(names.insert(iid, get_str(&m, 2).unwrap().to_string()).is_none(), "name interned twice");
                }
            }
        }
        if let Some(td) = get_msg(p, packet::TRACK_DESCRIPTOR) {
            let uuid = get_u64(&td, proto::track_descriptor::UUID).unwrap();
            let (name, is_thread) = if let Some(t) = get_msg(&td, proto::track_descriptor::THREAD) {
                (get_str(&t, proto::thread_descriptor::THREAD_NAME).unwrap().to_string(), true)
            } else if let Some(pr) = get_msg(&td, proto::track_descriptor::PROCESS) {
                (get_str(&pr, proto::process_descriptor::PROCESS_NAME).unwrap().to_string(), false)
            } else {
                (get_str(&td, proto::track_descriptor::NAME).unwrap().to_string(), false)
            };
            let counter = get(&td, proto::track_descriptor::COUNTER).is_some();
            assert!(d.tracks.insert(uuid, (name, is_thread, counter)).is_none(), "track described twice");
        }
        if let Some(ev) = get_msg(p, packet::TRACK_EVENT) {
            let track = get_u64(&ev, te::TRACK_UUID).unwrap();
            assert!(d.tracks.contains_key(&track), "event on undescribed track {track:#x}");
            let name = get_u64(&ev, te::NAME_IID).map(|i| names[&i].clone());
            let arg = get_msg(&ev, te::DEBUG_ANNOTATIONS).and_then(|a| get_u64(&a, proto::debug_annotation::UINT_VALUE));
            d.events.push((
                get_u64(p, packet::TIMESTAMP).unwrap(),
                get_u64(&ev, te::TYPE).unwrap(),
                track,
                name,
                get_u64(&ev, te::COUNTER_VALUE).map(|v| v as i64),
                arg,
            ));
        }
    }
    d
}

#[test]
fn session_round_trip() {
    let _g = SERIAL.lock();
    let path = temp_path("rt");
    start(&path, cat::ALL, Clock::Host).unwrap();
    assert!(start(&path, cat::ALL, Clock::Host).is_err(), "second session must be refused");

    {
        crate::pf_scope!(T_SLICE, 7);
        crate::pf_instant!(T_INSTANT);
    }
    std::thread::Builder::new()
        .name("pf-test-worker".into())
        .spawn(|| {
            crate::pf_begin!(T_SLICE, 1);
            crate::pf_end!(T_SLICE);
            crate::pf_instant!(T_INSTANT);
            crate::pf_counter!(T_COUNTER, -3);
        })
        .unwrap()
        .join()
        .unwrap();

    let msg = stop().unwrap();
    assert!(msg.contains("events"), "{msg}");
    assert!(stop().is_err());

    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let d = decode(&data);

    let user: Vec<_> = d.events.iter().filter(|e| e.3.as_deref() != Some("guest MIPS")
        && d.tracks[&e.2].0 != "guest MIPS" && d.tracks[&e.2].0 != "perfetto dropped events").collect();
    assert_eq!(user.len(), 7, "{user:?}");
    let ts: Vec<u64> = d.events.iter().map(|e| e.0).collect();
    assert!(ts.windows(2).all(|w| w[0] <= w[1]), "events not in time order");

    let begins: Vec<_> = user.iter().filter(|e| e.1 == te::TYPE_SLICE_BEGIN).collect();
    assert_eq!(begins.len(), 2);
    assert!(begins.iter().all(|e| e.3.as_deref() == Some("test slice")));
    let mut args: Vec<_> = begins.iter().map(|e| e.5.unwrap()).collect();
    args.sort();
    assert_eq!(args, [1, 7]);
    assert_eq!(user.iter().filter(|e| e.1 == te::TYPE_SLICE_END).count(), 2);

    // Both instants share the named track; slices are on their threads'.
    let instants: Vec<_> = user.iter().filter(|e| e.1 == te::TYPE_INSTANT).collect();
    assert_eq!(instants.len(), 2);
    assert_eq!(instants[0].2, instants[1].2);
    assert_eq!(d.tracks[&instants[0].2], ("test irq".to_string(), false, false));
    assert!(begins.iter().all(|e| d.tracks[&e.2].1), "slices belong on thread tracks");
    assert!(begins.iter().any(|e| d.tracks[&e.2].0 == "pf-test-worker"));

    let ctr = user.iter().find(|e| e.1 == te::TYPE_COUNTER).unwrap();
    assert_eq!(ctr.4, Some(-3));
    assert_eq!(d.tracks[&ctr.2], ("test counter".to_string(), false, true));
}

#[test]
fn category_mask_filters_and_stopped_records_nothing() {
    let _g = SERIAL.lock();
    crate::pf_instant!(T_GFX); // no session: must not record (or register a ring)
    let path = temp_path("mask");
    start(&path, cat::IRQ, Clock::Host).unwrap();
    crate::pf_instant!(T_GFX);
    crate::pf_instant!(T_INSTANT);
    set_categories(cat::GFX).unwrap();
    crate::pf_instant!(T_GFX);
    crate::pf_instant!(T_INSTANT);
    stop().unwrap();
    crate::pf_instant!(T_INSTANT);
    assert!(set_categories(cat::ALL).is_err());

    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let d = decode(&data);
    let names: Vec<_> = d.events.iter().filter_map(|e| e.3.clone()).collect();
    assert_eq!(names, ["test instant", "test gfx"]);
}

#[test]
fn guest_clock_uses_instruction_count() {
    let _g = SERIAL.lock();
    let counter: &'static mut u64 = Box::leak(Box::new(1000));
    let cp = counter as *mut u64;
    set_guest_cycles(Some(crate::cpu::mips_core::CyclesPtr::new(cp)));
    let path = temp_path("guest");
    start(&path, cat::ALL, Clock::Guest).unwrap();
    crate::pf_instant!(T_INSTANT);
    unsafe { std::ptr::write_volatile(cp, 5000) };
    crate::pf_instant!(T_INSTANT);
    stop().unwrap();
    set_guest_cycles(None);

    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let d = decode(&data);
    let ts: Vec<u64> = d.events.iter().filter(|e| e.1 == te::TYPE_INSTANT).map(|e| e.0).collect();
    assert_eq!(ts, [1000, 5000]);
}

#[test]
fn full_ring_drops_and_counts() {
    let ring = Ring::new(0, 0, "t".into());
    let r = Record { def: &T_GFX, host_ticks: 0, guest_kind: Kind::Instant as u64, arg: 0 };
    let end = Record { guest_kind: Kind::End as u64, ..r };
    // Instants stop at 7/8 full; begins and ends may use the rest.
    for _ in 0..RING_SLOTS {
        ring.push(r);
    }
    assert_eq!(ring.take_dropped(), (RING_SLOTS / 8) as u64);
    for _ in 0..RING_SLOTS / 8 + 5 {
        ring.push(end);
    }
    assert_eq!(ring.take_dropped(), 5);
    let mut out = Vec::new();
    ring.drain_into(&mut out);
    assert_eq!(out.len(), RING_SLOTS);
    // Space again after the drain.
    ring.push(r);
    assert_eq!(ring.take_dropped(), 0);
    out.clear();
    ring.drain_into(&mut out);
    assert_eq!(out.len(), 1);
}

#[test]
fn parse_args() {
    assert_eq!(cat::parse("all,-jit").unwrap(), cat::ALL & !cat::JIT);
    assert_eq!(cat::parse("irq, gfx").unwrap(), cat::IRQ | cat::GFX);
    assert!(cat::parse("bogus").is_err());
    let (p, c, k) = parse_start_args(&["x.pftrace", "cats=scsi", "clock=guest"]).unwrap();
    assert_eq!((p.as_str(), c, k), ("x.pftrace", cat::SCSI, Clock::Guest));
    assert!(parse_start_args(&[]).is_err());
    assert!(parse_start_args(&["x", "nope"]).is_err());
}

/// Writes a small multi-thread trace to `$IRIS_PERFETTO_DEMO` for checking
/// with the real trace processor or ui.perfetto.dev:
/// `IRIS_PERFETTO_DEMO=/tmp/demo.pftrace cargo test --features perfetto -- --ignored demo_trace`
#[test]
#[ignore]
fn demo_trace() {
    let _g = SERIAL.lock();
    let path = std::env::var("IRIS_PERFETTO_DEMO").expect("set IRIS_PERFETTO_DEMO");
    static OUTER: EventDef = EventDef::new("outer", cat::CPU).with_arg("i");
    static INNER: EventDef = EventDef::new("inner", cat::JIT);
    static DMA: EventDef = EventDef::new("dma busy", cat::DMA).on_track("DMA channel");
    static DEPTH: EventDef = EventDef::new("fifo depth", cat::GFX);
    start(&path, cat::ALL, Clock::Host).unwrap();
    let workers: Vec<_> = (0..3)
        .map(|t| {
            std::thread::Builder::new()
                .name(format!("demo-{t}"))
                .spawn(move || {
                    for i in 0..200u64 {
                        crate::pf_scope!(OUTER, i);
                        {
                            crate::pf_scope!(INNER);
                            std::thread::sleep(std::time::Duration::from_micros(50));
                        }
                        if t == 0 {
                            if i % 2 == 0 { crate::pf_begin!(DMA) } else { crate::pf_end!(DMA) }
                            crate::pf_counter!(DEPTH, (i * 7) % 32);
                        } else if t == 1 && i % 2 == 0 {
                            crate::pf_instant!(T_INSTANT, i);
                        }
                    }
                })
                .unwrap()
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    eprintln!("{}", stop().unwrap());
}

/// Cost of an emitted and of a filtered-out event:
/// `cargo test --release --features perfetto -- --ignored emit_cost --nocapture`
#[test]
#[ignore]
fn emit_cost() {
    let _g = SERIAL.lock();
    const N: u64 = 2_000_000;
    let path = temp_path("cost");
    start(&path, cat::IRQ, Clock::Host).unwrap();
    let t = std::time::Instant::now();
    for i in 0..N {
        crate::pf_instant!(T_GFX, i); // filtered out
    }
    let off = t.elapsed().as_nanos() as f64 / N as f64;
    // Chunks of half a ring, with a pause after each for the writer to
    // drain, so this measures the emit path rather than overflow.
    let mut busy = std::time::Duration::ZERO;
    for chunk in 0..N / 32768 {
        let t = std::time::Instant::now();
        for i in 0..32768 {
            crate::pf_instant!(T_INSTANT, chunk * 32768 + i);
        }
        busy += t.elapsed();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let on = busy.as_nanos() as f64 / (N / 32768 * 32768) as f64;
    let msg = stop().unwrap();
    let _ = std::fs::remove_file(&path);
    eprintln!("filtered: {off:.2} ns/event  recorded: {on:.1} ns/event  ({msg})");
}

static T_LINES: [EventDef; 32] = {
    const L: EventDef = EventDef::new("line", cat::IRQ).on_track("line").in_group("test group");
    let mut a = [L; 32];
    a[0] = EventDef::new("line0", cat::IRQ).on_track("line0").in_group("test group");
    a[1] = EventDef::new("line1", cat::IRQ).on_track("line1").in_group("test group");
    a
};
static T_LEVEL: EventDef = EventDef::new("level", cat::GFX).on_track("level track");
static T_VALUE: EventDef = EventDef::new("value", cat::GFX);

#[test]
fn edge_trackers_survive_session_restarts() {
    let _g = SERIAL.lock();
    let level = crate::PfLevel::new();
    let bits = crate::PfBits::new();
    let value = crate::PfValue::new();

    // Before any session: nothing is recorded, nothing is remembered.
    crate::pf_level!(level, T_LEVEL, true);
    crate::pf_bits!(bits, T_LINES, 0b01);

    let path = temp_path("edges1");
    start(&path, cat::ALL, Clock::Host).unwrap();
    crate::pf_level!(level, T_LEVEL, true);   // begin (opened in this session)
    crate::pf_level!(level, T_LEVEL, true);   // no-op
    crate::pf_bits!(bits, T_LINES, 0b01);     // line0 already high: begin now
    crate::pf_bits!(bits, T_LINES, 0b10);     // line0 end, line1 begin
    crate::pf_value!(value, T_VALUE, 5);
    crate::pf_value!(value, T_VALUE, 5);      // unchanged: nothing
    crate::pf_value!(value, T_VALUE, 7);
    stop().unwrap();
    let d = decode(&std::fs::read(&path).unwrap());
    let _ = std::fs::remove_file(&path);
    let ev: Vec<(u64, String)> = d.events.iter()
        .filter(|e| d.tracks[&e.2].0 != "guest MIPS")
        .map(|e| (e.1, d.tracks[&e.2].0.clone())).collect();
    assert_eq!(ev, [
        (te::TYPE_SLICE_BEGIN, "level track".into()),
        (te::TYPE_SLICE_BEGIN, "line0".into()),
        (te::TYPE_SLICE_END, "line0".into()),
        (te::TYPE_SLICE_BEGIN, "line1".into()),
        (te::TYPE_COUNTER, "value".into()),
        (te::TYPE_COUNTER, "value".into()),
    ]);
    // The line tracks hang off their group, which hangs off the process.
    assert!(d.tracks.values().any(|t| t.0 == "test group"));

    // Second session with the level and line1 still high: the level's
    // falling edge must not emit an end for a slice this session never
    // opened; line1 is re-opened at the first update.
    let path = temp_path("edges2");
    start(&path, cat::ALL, Clock::Host).unwrap();
    crate::pf_level!(level, T_LEVEL, false);
    crate::pf_bits!(bits, T_LINES, 0b10);
    crate::pf_value!(value, T_VALUE, 7);      // new session: emitted again
    stop().unwrap();
    let d = decode(&std::fs::read(&path).unwrap());
    let _ = std::fs::remove_file(&path);
    let ev: Vec<(u64, String)> = d.events.iter()
        .filter(|e| d.tracks[&e.2].0 != "guest MIPS")
        .map(|e| (e.1, d.tracks[&e.2].0.clone())).collect();
    assert_eq!(ev, [
        (te::TYPE_SLICE_BEGIN, "line1".into()),
        (te::TYPE_COUNTER, "value".into()),
    ]);
}
