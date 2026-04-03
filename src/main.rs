use iris::config::load_config;
use iris::machine::Machine;
use iris::ui::Ui;
use winit::event_loop::EventLoop;

fn main() {
    let (cfg, scale) = load_config();
    let event_loop = EventLoop::new().unwrap();
    // Machine::new() allocates >1MB on the stack (Physical device_map), which overflows
    // the default stack on Windows (1MB). We spawn a thread with a larger stack to create it.
    let mut machine = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || Box::new(Machine::new(cfg)))
        .unwrap()
        .join()
        .unwrap();
    machine.register_system_controller();
    machine.start();
    std::thread::spawn(|| {
        Machine::run_console_client();
    });

    let ui = Ui::new(machine.get_ps2(), machine.get_rex3(), machine.get_timer_manager(), &event_loop, scale);
    ui.run(event_loop);
    
    machine.stop();
}
