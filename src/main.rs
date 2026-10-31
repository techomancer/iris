use iris::config::load_config;
use iris::machine::Machine;

fn main() {
    // Without a backend every `log::warn!` here is discarded. Stays at
    // `error` unless RUST_LOG is set, as iris-gui already does.
    env_logger::init();

    // Turn a silent death (exit 0xC000041D and friends) into iris-crash.log.
    // Cheap: dormant until something actually crashes. See the module docs.
    iris::crash_diag::install();
    if let Ok(kind) = std::env::var("IRIS_CRASH_SELFTEST") {
        iris::crash_diag::selftest(&kind);
    }
    print_build_features();

    // `mut` is only used by the pcap interface prompt below.
    #[cfg_attr(not(feature = "pcap"), allow(unused_mut))]
    let (mut cfg, scale) = load_config();

    // If PCAP networking is selected but no interface was configured, prompt the
    // user to choose one now — on the main thread, before the machine starts —
    // and write the choice into the config. Done here (not in the network
    // backend, which spins up on its own thread during boot) so the prompt
    // actually gates startup and reliably reads the console on every platform.
    // Skipped silently when there's no interactive console (headless/CI), where
    // the backend then auto-picks.
    #[cfg(feature = "pcap")]
    {
        if cfg.network.mode == iris::config::NetMode::Pcap
            && cfg.network.pcap_interface.as_deref().map(str::trim).unwrap_or("").is_empty()
        {
            if let Some(name) = iris::net::net_pcap::prompt_for_interface() {
                cfg.network.pcap_interface = Some(name);
            }
        }
    }

    let scroll_pixels_per_line = cfg.mouse_scroll_pixels_per_line;
    let lock_aspect_ratio = cfg.lock_aspect_ratio;
    let headless = cfg.headless;
    let gdb_port = cfg.gdb_port;
    let ci_enabled = cfg.ci;
    let ci_display = cfg.ci_display;
    let ci_socket_path = cfg.ci_socket.clone();
    let load_elf = cfg.load_elf.clone();
    let vnc = cfg.vnc.clone();
    let perfetto = (cfg.perfetto.clone(), cfg.perfetto_cats.clone(), cfg.perfetto_clock.clone());

    // Apply [debug] and [jitv2] from TOML (env vars still override if set externally).
    cfg.debug.apply_env();
    cfg.jitv2.apply_env();

    // Machine::new() has several large temporaries on the stack, which
    // overflows the default stack on Windows (1MB). We spawn a thread with a
    // larger stack to create it.
    let mut machine = std::thread::Builder::new()
        // Rex3Context embeds the HOSTRW data-port array (HOSTRW_BUF_QWORDS u64s,
        // 1 MiB), and construction has several context-sized temporaries in
        // flight before the machine is boxed. (The physical decode table used
        // to be another 1 MB here; it is host-mapped now.) This is virtual
        // address space, lazily committed, so the large reservation has no
        // real cost.
        .stack_size(64 * 1024 * 1024)
        .spawn(move || iris::workstation::build(cfg))
        .unwrap()
        .join()
        .unwrap()
        .unwrap_or_else(|e| {
            eprintln!("iris: {e}");
            std::process::exit(1);
        });
    machine.register_system_controller();

    // --load-elf: load before the CPU starts, so the binary runs instead of the
    // PROM's boot path. RAM is already mapped at this point (Machine::new fires
    // an initial remap_banks with the MC's power-on MEMCFG), so this lands in
    // real RAM without POST having run.
    if let Some(path) = &load_elf {
        match machine.load_elf(path) {
            Ok(summary) => eprint!("iris: loaded {}\n{}", path, summary),
            Err(e) => {
                eprintln!("iris: --load-elf {}: {}", path, e);
                std::process::exit(1);
            }
        }
    }

    // DIAG: optionally enable verbose logging from startup via IRIS_DEBUG_LOG.
    // IRIS_DEBUG_LOG="mc,mips" enables those modules. "all" enables everything.
    // Output is broadcast to a stderr sink so jit-diag.sh's tee captures it inline.
    if let Ok(spec) = std::env::var("IRIS_DEBUG_LOG") {
        if let Some(dl) = iris::devlog::DEVLOG.get() {
            // Register stderr as a sink so dlog output reaches our captured log.
            let stderr_sink: iris::devlog::DevLogWriter = std::sync::Arc::new(
                parking_lot::Mutex::new(std::io::stderr()),
            );
            dl.add_sink(stderr_sink);

            for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if name == "all" {
                    for m in iris::devlog::LogModule::all() { dl.enable(*m); }
                    eprintln!("DIAG: enabled all log modules -> stderr");
                } else if let Some(m) = iris::devlog::LogModule::from_str(name) {
                    dl.enable(m);
                    eprintln!("DIAG: enabled log module {} -> stderr", m.name());
                } else {
                    eprintln!("DIAG: unknown log module '{}'", name);
                }
            }
        }
    }

    // Start GDB stub before starting the CPU so that in developer mode (CPU not
    // auto-started), GDB can connect and set breakpoints before running.
    if let Some(port) = gdb_port {
        let cpu_debug = machine.get_cpu_debug();
        iris::gdb_stub::start_gdb_server(port, cpu_debug);
    }

    start_perfetto(perfetto);

    machine.start();

    // CI control socket: started after the machine so a successful connection
    // means the monitor has already bound its port. That makes "the CI socket
    // answers" a reliable readiness signal for a harness — no fixed sleep —
    // and keeps CI commands from racing the half-started Machine. The CPU
    // itself stays parked in `--ci` mode (see Machine::start).
    let _ci_server = if ci_enabled {
        match iris::ci::start_server(&mut *machine, &ci_socket_path) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("iris: failed to start CI server: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // The monitor's `screen`, `ptr` and `key` commands, and the VNC server's
    // hands on the guest.
    let host_input = iris::hostinput::HostInput::attach(&*machine);
    let _vnc_server = vnc.listen.as_deref().map(|listen| {
        iris::vnc::VncServer::start(host_input.clone(), listen, vnc.max_fps, "IRIS")
            .unwrap_or_else(|e| {
                eprintln!("iris: {e}");
                std::process::exit(1);
            })
    });

    // The terminal console talks to this process's own monitor only. Without
    // one (its port taken by another iris), there is no console: attached to
    // the other iris's monitor, it would exit along with that iris.
    if !ci_enabled {
        if let Some(addr) = machine.monitor_addr() {
            let slot = machine.machine_slot();
            std::thread::spawn(move || {
                Machine::run_console_client(addr, iris::workstation::console_quit(slot));
            });
        }
    }

    // A machine with no graphics board (an Octane without one) runs on its
    // serial console, as headless does.
    let no_board = machine.get_display().is_none();
    let show_window = !headless && !no_board && !(ci_enabled && !ci_display);
    if !show_window {
        if headless {
            eprintln!("iris: running headless (no REX3, no window)");
        } else if no_board {
            eprintln!("iris: no graphics board fitted: running on the serial console, no window");
        } else if ci_enabled {
            eprintln!("iris: --ci mode (REX3 rendering to offscreen buffer, no window)");
        }
        // Park the main thread so background threads (CPU, REX3 refresh,
        // CI socket) keep running. `quit` via the CI socket calls
        // std::process::exit.
        std::thread::park();
    } else {
        use iris::ui::Ui;
        use winit::event_loop::EventLoop;
        let event_loop = EventLoop::new().unwrap();
        let display = machine.get_display().expect("a graphics board must be present in non-headless mode");
        let ui = Ui::new(machine.get_ps2(), display, machine.disc_changer(), machine.get_timer_manager(), &event_loop, scale, scroll_pixels_per_line, lock_aspect_ratio);
        ui.run(event_loop);
    }

    let _ = iris::workstation::stop_and_flush(&mut *machine, "exit");
    #[cfg(feature = "perfetto")]
    iris::perfetto::shutdown();
}

/// `--perfetto FILE [--perfetto-cats LIST] [--perfetto-clock CLOCK]`.
fn start_perfetto((path, cats, clock): (Option<String>, Option<String>, Option<String>)) {
    let Some(path) = path else {
        if cats.is_some() || clock.is_some() {
            eprintln!("iris: --perfetto-cats/--perfetto-clock need --perfetto FILE");
        }
        return;
    };
    #[cfg(feature = "perfetto")]
    {
        use iris::perfetto;
        let res = (|| {
            let cats = cats.as_deref().map_or(Ok(perfetto::cat::ALL), perfetto::cat::parse)?;
            let clock = clock.as_deref().map_or(Ok(perfetto::Clock::Host), perfetto::Clock::parse)?;
            perfetto::start(&path, cats, clock)?;
            Ok::<_, String>(format!("iris: perfetto recording to {path} (clock={}, cats={})",
                clock.name(), perfetto::cat::format(cats)))
        })();
        match res {
            Ok(msg) => eprintln!("{msg}"),
            Err(e) => {
                eprintln!("iris: {e}");
                std::process::exit(1);
            }
        }
    }
    #[cfg(not(feature = "perfetto"))]
    {
        eprintln!("iris: --perfetto {path}: this build has no perfetto support (rebuild with --features perfetto)");
        std::process::exit(1);
    }
}

/// Print which compile-time feature flags this binary was built with. Handy
/// when diagnosing behaviour that depends on the build (e.g. MIPS `jit` bypasses
/// the interpreter's idle-park path, so an idle guest spins the host CPU).
/// The list itself lives in the library — see `iris::build_features::enabled`.
fn print_build_features() {
    eprintln!("iris: build features: {}", iris::build_features::banner());
}

