# Profile menus must use the core hardware tables and platform storage rules

Status: current (2026-11-12).

Symptom: configuration supported twelve profiles but the GUI offered five;
Octane omitted VPro, and RAM selectors stopped after bank 3. Merely expanding
the dropdowns also exposed Indy-specific NVRAM seeding, EEPROM MAC editing,
channel-B console routing, and missing platform/expansion path traversal.

Rules:

- `MachineProfile::ALL` aliases `EVERY`. Use `cpus()`, `memory_banks()` and
  `bank_sizes()` when drawing hardware controls, including the menu bar and
  new-machine dialog. Processor-count constraints also depend on the Tezro chip.
- Offer graphics from `GraphicsBoard::ALL`, filtered by hardware. O2 has fixed
  GBE, not an interchangeable GIO board. Onyx2 does not take IR3 even though
  the core's broad `supports()` branch currently includes every KONA product.
- Refit CPU, bank layout and graphics only when a profile changes. Do not
  normalize on every frame: it destroys deliberate settings.
- Indy/Indigo2 battery seeding and EEPROM MAC operations must not touch O2,
  SN0 or IP35 stores. NVRAM reset is available only for MC profiles and Octane.
- IP35 bricks retain the core's default SCSI map and attach disks through
  Fibre Channel. Do not preflight those unused SCSI defaults as real disks.
- Add every backing-file/socket path to `machines::visit_paths`; relative
  saving, migration and sandbox bookmark harvesting all depend on this walk.
- `serial_console::console_port` follows the platform's selected console;
  channel B is correct only for the Indy/Indigo2 family.
- `MachineConfig::perfetto*` are `serde(skip)` process launch fields, not
  persistable machine settings. The GUI's recording controls own trace capture.

Regression coverage: `extra_ui::tests`, the machine path round-trip tests and
`serial_console::routing_tests`. GUI tests need shared-memory access for the
emulator lifecycle child. Egui 0.36 headless frames use `Context::run_ui` and
must call `drop_without_applying_deltas` when not presenting the output.
