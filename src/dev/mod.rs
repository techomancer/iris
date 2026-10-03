//! Emulated hardware. Graphics boards get a directory each (GR2, Newport,
//! IMPACT); the rest of the system's devices sit here as single files.

pub mod gl;
pub mod gr2;
pub mod mgras;
pub mod ng1;

pub mod camera;
pub mod cdmc;
pub mod daynaport;
pub mod ds1x86;
pub mod eeprom_93c56;
pub mod hal2;
pub mod hpc3;
pub mod ioc;
pub mod mc;
pub mod mc_vdma;
pub mod mem;
pub mod pit8254;
pub mod prom;
pub mod ps2;
pub mod saa7191;
pub mod seeq8003;
pub mod testdev;
pub mod timer;
pub mod ultra64;
pub mod vino;
pub mod wd33c93a;
pub mod z85c30;
