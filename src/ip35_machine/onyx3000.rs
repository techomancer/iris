//! The Onyx 3000: the Origin 3200's C-brick and I-brick ([`super::origin3200`]) with graphics.
//!
//! Selected with `[machine] profile = "onyx3000_ip35"`; the shared keys and `[onyx3000]` (the keys of
//! `[origin3200]`; `c_bricks` is refused) apply as there, and `[graphics] board` picks the graphics
//! brick. Only ir3's widget 11 is established; the rest is the model's composition.
//!
//! `board = "ir3"` is one C-brick, the I-brick and a single-pipe G-brick (an Onyx 3200): the pipe
//! sits at widget 11 of the I-brick's XBridge ([`SPARE_PORT`]), its DMA and interrupts reaching the
//! C-brick through the XBridge's host port. `board = "v12"` is two C-bricks joined by NUMAlink,
//! the first with the I-brick, the second with a V-brick (an XBridge and the V12 at widget 8) on its
//! own XIO connector; the V-brick cabling to a C-brick is established, its having no PCI devices inferred.
//!
//! # What is approximated here
//!
//! - The V12's widget: 8 and 12 are equal candidates from the X-brick's slot table; 8 is the choice
//!   ([`V12_WIDGET`]). The V-brick's XBridge has both PCI heads, empty, as the chip always does.
//! - The V-brick's controller answers as an X-brick ([`V_BRICK`]): the type letter, and a stub serving
//!   every other request as its C-brick's; its real board records are not established.
//! - The G-brick's L1 and cable are not modelled; the pipe is a widget of the I-brick's XBridge
//!   (established: each pipe sits at widget 11 of an I-brick or X-brick XBridge over a direct cable;
//!   one behind an N-brick is not modelled).
//! - The I-brick's controller lists the G-brick at port 2 of its attached-brick reply, letter `G`, at
//!   [`G_BRICK_BAY`] (the model's choice; the PROM needs a name there or takes the port for a second host).
//! - The status bar's activity bits and clock blink are not wired: the pipe's status has its own counters.
//! - Which products an Onyx 3000 takes follows `KonaHost::Onyx3000`: InfiniteReality3 only on a G-brick.
//! - The I-brick has no PS/2 path (observed in the PROM run), so without a USB keyboard in its OHCI the
//!   PROM prints "Cannot connect to keyboard" and the board draws nothing; with one, `console = g` works.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c in jamesbraid/qemu-sgi-origami, where no
//! Onyx 3000 hardware graph has been captured.

use super::origin3200::{c_bricks, wire_ibrick, PortReset, I_BRICK, SPARE_PORT};
use crate::ip35::{odyssey_packets, xbridge_on};
use super::{BoardIo, NodeSpec};
use crate::hptimer::TimerManager;
use crate::config::MachineConfig;
use crate::dev::bedrock::Bedrock;
use crate::dev::kona::{Kona, KonaConfig};
use crate::dev::l1::AttachedBrick;
use crate::dev::odyssey::{Odyssey, PacketSender};
use crate::dev::xbridge::{XBridge, HEAD_E, HEAD_F, HOST_WIDGET};
use crate::gfx_display::GfxDisplay;
use crate::dev::bedrock_flash::Flash;
use crate::dev::l1::L1;
use crate::sn_nodes::NodeBoard;
use crate::irq::InterruptController;
use crate::traits::{BusDevice, Device, Resettable};
use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock, Weak};

/// Where an InfiniteReality pipe sits: widget 11 of the I-brick's XBridge,
/// its port B.
const PIPE_WIDGET: u8 = SPARE_PORT;

/// The rack bay the I-brick's controller reports for the G-brick: a choice, as no inventory places
/// one (the C-brick is at bay 4, the I-brick at 14).
const G_BRICK_BAY: u32 = 24;

/// The V-brick's V12 at widget 8 of its XBridge. Approximation: no V-brick widget listing is
/// captured; 8 and 12 are equal candidates from the X-brick's slot table.
const V12_WIDGET: u8 = 8;

/// The V-brick as its C-brick's controller reports it. Approximation: the L1 firmware names none;
/// it runs the X-brick's personality, so the type letter is `X`, the bay a choice and the rest the
/// C-brick's.
const V_BRICK: AttachedBrick = AttachedBrick { bay: 11, type_code: b'X' as u32, graphics_bay: None };

/// What the graphics brick adds to the I-brick's I/O: its window and status source, its monitor
/// entry, and what start, stop, reset and locking do to it.
struct Graphics {
    display: Arc<dyn GfxDisplay>,
    status: Arc<crate::disp::StatusSource>,
    monitor: Arc<dyn Device>,
    start: Box<dyn Fn() + Send + Sync>,
    stop: Box<dyn Fn() + Send + Sync>,
    reset: Box<dyn Fn() + Send + Sync>,
    locks: Box<dyn Fn() + Send + Sync>,
}

impl Graphics {
    /// Join the board to the I/O. The board stops before the I/O it sits
    /// behind and starts after it.
    fn join(self, io: &mut BoardIo) {
        io.display = Some(self.display);
        io.status = Some(self.status);
        io.monitor.push(self.monitor);
        let (start, stop, reset, locks) = (
            std::mem::replace(&mut io.start, Box::new(|| {})),
            std::mem::replace(&mut io.stop, Box::new(|| {})),
            std::mem::replace(&mut io.reset, Box::new(|| {})),
            std::mem::replace(&mut io.register_locks, Box::new(|| {})),
        );
        let (board_start, board_stop, board_reset, board_locks) = (self.start, self.stop, self.reset, self.locks);
        io.start = Box::new(move || {
            start();
            board_start();
        });
        io.stop = Box::new(move || {
            board_stop();
            stop();
        });
        io.reset = Box::new(move || {
            reset();
            board_reset();
        });
        io.register_locks = Box::new(move || {
            locks();
            board_locks();
        });
    }
}

fn new_status() -> Arc<crate::disp::StatusSource> {
    crate::disp::StatusSource::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)))
}

/// The Onyx 3000's C-bricks: one with a G-brick, or two with a V-brick.
pub(super) fn specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
    if cfg.graphics_board().is_vpro() {
        c_bricks(cfg, 2, I_BRICK, Some(V_BRICK))
    } else {
        c_bricks(cfg, 1, AttachedBrick { graphics_bay: Some(G_BRICK_BAY), ..I_BRICK }, None)
    }
}

/// The I-brick on the first C-brick and the graphics brick behind it.
pub(super) fn wire(cfg: &MachineConfig, nodes: &[NodeBoard<Bedrock, Flash>], l1s: &[Arc<L1>], _: &Arc<TimerManager>) -> Result<BoardIo, String> {
    if cfg.graphics_board().is_vpro() {
        let (mut io, _) = wire_ibrick(cfg, nodes, l1s, None)?;
        let v_node = nodes.get(1).ok_or("the V-brick needs a C-brick of its own")?;
        wire_vbrick(v_node, cfg)?.join(&mut io);
        return Ok(io);
    }
    let (product, rm_count) = cfg
        .graphics_board()
        .kona()
        .ok_or_else(|| format!("graphics.board \"{}\" is not a board of the Onyx 3000", cfg.graphics_board().as_str()))?;
    // The pipe is built after the crossbar it sits on, and a link reset
    // of its port resets the widget's Crosstalk registers, not the board.
    let cell: Arc<OnceLock<Weak<Kona>>> = Arc::new(OnceLock::new());
    let reset_cell = cell.clone();
    let port_reset: PortReset = Arc::new(move |port| {
        if port == PIPE_WIDGET {
            if let Some(pipe) = reset_cell.get().and_then(Weak::upgrade) {
                pipe.link_reset();
            }
        }
    });
    let (mut io, xbridge) = wire_ibrick(cfg, nodes, l1s, Some(port_reset))?;
    let status = new_status();
    let pipe = fit_pipe(&nodes[0].chip, &xbridge, cfg, KonaConfig::new(product, rm_count)?, &status)?;
    let _ = cell.set(Arc::downgrade(&pipe));
    eprintln!("iris: {} on the G-brick at I-brick widget {PIPE_WIDGET}", product.spec().label);
    let (start, stop, reset) = (pipe.clone(), pipe.clone(), pipe.clone());
    Graphics {
        display: pipe.display(),
        status,
        monitor: pipe,
        start: Box::new(move || start.start()),
        stop: Box::new(move || stop.stop()),
        reset: Box::new(move || reset.power_on()),
        locks: Box::new(|| {}),
    }
    .join(&mut io);
    Ok(io)
}

/// A V-brick on `node`'s XIO connector: an XBridge with no PCI devices and the V12 at
/// [`V12_WIDGET`], its DMA and interrupts going through the node's Bedrock as any widget's do.
fn wire_vbrick(node: &NodeBoard<Bedrock, Flash>, cfg: &MachineConfig) -> Result<Graphics, String> {
    let bedrock = &node.chip;
    let xbridge = xbridge_on(bedrock);
    for widget in [0, HOST_WIDGET, HEAD_E, HEAD_F] {
        bedrock.io.connect_widget(widget, xbridge.widget(widget).expect("the V-brick's ports are the XBridge's"));
    }
    let status = new_status();
    let packets = odyssey_packets(bedrock, V12_WIDGET);
    let board = cfg.graphics_board();
    let v12 = Odyssey::new_widget_relative(
        V12_WIDGET,
        board.as_str(),
        packets,
        bedrock.inbound(V12_WIDGET),
        status.clone(),
        (cfg.graphics.threads, cfg.graphics.tile),
        (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str()),
    )?;
    bedrock.io.connect_widget(V12_WIDGET, xbridge.attach_widget(V12_WIDGET, v12.clone()));
    // A link reset of the V12's port resets the board behind it.
    let reset_v12 = Arc::downgrade(&v12);
    xbridge.set_reset_output(Arc::new(move |port| {
        if port == V12_WIDGET {
            if let Some(v12) = reset_v12.upgrade() {
                v12.power_on();
            }
        }
    }));
    eprintln!("iris: {} on the V-brick at widget {V12_WIDGET}", board.label());
    let (start, stop, reset_all, locks) = (v12.clone(), v12.clone(), v12.clone(), xbridge.clone());
    let reset_bridge = xbridge.clone();
    Ok(Graphics {
        display: v12.display(),
        status,
        monitor: v12,
        start: Box::new(move || start.start()),
        stop: Box::new(move || stop.stop()),
        reset: Box::new(move || {
            reset_bridge.power_on();
            reset_all.power_on();
        }),
        locks: Box::new(move || locks.register_locks()),
    })
}

/// An InfiniteReality pipe at widget 11 of the I-brick's XBridge: DMA enters Bedrock as that
/// widget's inbound traffic, interrupt packets go to the address the driver programs, defaulting to
/// the host port.
fn fit_pipe(
    bedrock: &Arc<Bedrock>,
    xbridge: &Arc<XBridge>,
    cfg: &MachineConfig,
    kona: KonaConfig,
    status: &Arc<crate::disp::StatusSource>,
) -> Result<Arc<Kona>, String> {
    let kona = kona.with_host_widget(HOST_WIDGET)?.with_widget(PIPE_WIDGET)?.with_pool(cfg.graphics.threads, cfg.graphics.tile);
    let flash = cfg.graphics.flash.as_deref().map(|p| crate::dev::kona::flash_file(Path::new(p))).transpose()?;
    // Its interrupt packets are the chip's inbound writes from that widget, to
    // the address the driver programmed.
    let pipe = Kona::new(kona, bedrock.inbound(PIPE_WIDGET), bedrock.inbound(PIPE_WIDGET), status.clone(), flash)?;
    bedrock.io.connect_widget(PIPE_WIDGET, xbridge.attach_widget(PIPE_WIDGET, pipe.clone()));
    Ok(pipe)
}

#[cfg(test)]
mod tests;
