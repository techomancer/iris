//! The Onyx 3000: the Origin 3200's C-brick and I-brick ([`super::origin3200`])
//! with graphics.
//!
//! Selected with `[machine] profile = "onyx3000_ip35"`. The shared keys and
//! `[onyx3000]` (the keys of `[origin3200]` less `c_bricks`) apply as on the
//! Origin 3200; `[graphics] board` picks the graphics brick and so the
//! composition. Only ir3's widget 11 is established; the rest is the model's
//! composition.
//!
//! # The G-brick
//!
//! `board = "ir3"` is one C-brick, the I-brick and one single-pipe G-brick: the
//! smallest real Onyx 3000, an Onyx 3200. The pipe sits at widget 11 of the
//! I-brick's XBridge, the port the row declares empty on the Origin 3200
//! ([`IBRICK`]`.spare_port`), and its DMA and interrupt packets reach the
//! C-brick through the XBridge's host port as any widget's do.
//!
//! # The V-brick
//!
//! `board = "v12"` is two C-bricks joined by the NUMAlink cable: the first with
//! the I-brick, the second with a V-brick on its own XIO connector. The V-brick
//! holds an XBridge and the VPro V12 at widget 8. Established: the V-brick
//! holds an XBridge and cables to a C-brick of its own. Inferred: no PCI
//! devices.
//!
//! # What is approximated here
//!
//! - The V12's widget: 8 and 12 are equal candidates from the X-brick's slot
//!   table, and 8 is the choice ([`V12_WIDGET`]).
//! - The V-brick's controller answers as an X-brick ([`V_BRICK`]): the type
//!   letter, and a stub that serves every other request as its C-brick's. Which
//!   board records a V-brick's controller answers with is not established;
//!   the X-brick's are an inferred candidate.
//! - The V-brick's XBridge has both PCI heads, empty, as the chip always does.
//! - The G-brick's own L1 and its cable are not modelled: the pipe is a
//!   widget of the I-brick's XBridge. Established is that each InfiniteReality
//!   pipe sits at widget 11 of an I-brick or X-brick XBridge over a direct
//!   cable with no router; a pipe behind an N-brick is not modelled.
//! - The I-brick's controller lists the G-brick at port 2 of its
//!   attached-brick reply, letter `G`, at rack bay [`G_BRICK_BAY`]. The PROM
//!   needs a name for the brick on the live second port, or it takes the port
//!   for a second host. The port, letter and bay are the model's choice.
//! - The status bar's activity bits and processor-clock blink are not wired:
//!   the pipe's status source has its own counters.
//! - Which products an Onyx 3000 takes follows the pipe's product table
//!   (`KonaHost::Onyx3000`): InfiniteReality3 only on a G-brick.
//! - The graphics console needs a keyboard, and the I-brick's is USB: it has
//!   no PS/2 keyboard path, and the PROM's attempt to open one fails (observed
//!   in the PROM run). Without a USB keyboard function in the I-brick's OHCI
//!   the PROM prints "Cannot connect to keyboard", keeps the menu on the
//!   serial console and the board draws nothing. With one fitted, `console = g` draws the PROM menu on the pipe.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c in
//! jamesbraid/qemu-sgi-origami, where no Onyx 3000 hardware graph has been
//! captured: widget 11 is established; the rest is the model's composition.

use super::origin3200::{c_bricks, wire_ibrick, PortReset, IBRICK, I_BRICK};
use super::{BoardIo, Ip35Board, NodeSpec};
use crate::config::MachineConfig;
use crate::dev::bedrock::Bedrock;
use crate::dev::kona::{Kona, KonaConfig};
use crate::dev::l1::AttachedBrick;
use crate::dev::odyssey::{Odyssey, PacketSender};
use crate::dev::xbridge::{XBridge, HEAD_E, HEAD_F, HOST_WIDGET};
use crate::gfx_display::GfxDisplay;
use crate::ip35::inbound::{Inbound, BEDROCK_PORT};
use crate::ip35::node::Node;
use crate::irq::InterruptController;
use crate::traits::{BusDevice, Device, Resettable};
use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock, Weak};

pub(crate) struct Onyx3000;

/// Where an InfiniteReality pipe sits: widget 11 of the I-brick's XBridge,
/// its port B.
const PIPE_WIDGET: u8 = IBRICK.spare_port;

/// The rack bay the I-brick's controller reports for the G-brick. A choice:
/// no Onyx 3000 inventory places one. The C-brick is at bay 4 and the I-brick
/// at 14; an 18 U G-brick fits above them.
const G_BRICK_BAY: u32 = 24;

/// The V-brick's V12 sits at widget 8 of its XBridge. Approximation: no
/// V-brick widget listing is captured. Widgets 8 and 12 are equal candidates,
/// from the X-brick's slot table if the V-brick's midplane matches it; 8 is
/// the model's choice.
const V12_WIDGET: u8 = 8;

/// The V-brick as its C-brick's controller reports it. Approximation: the L1
/// firmware names no V-brick; a V-brick runs the X-brick's personality, so
/// the controller answers the type letter `X` and the bay is a choice. The
/// V-brick's own controller is a stub, as the I-brick's is: everything past
/// the identity and the attached-brick list is served as the C-brick's.
const V_BRICK: AttachedBrick = AttachedBrick { bay: 11, type_code: b'X' as u32, graphics_bay: None };

/// What the graphics brick adds to the I/O the I-brick wiring built: the
/// board's window and status source, its place on the monitor, and what
/// starting, stopping, resetting and locking the machine does to it.
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

impl Ip35Board for Onyx3000 {
    const NAME: &'static str = "Onyx 3000";

    fn specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
        if cfg.graphics_board().is_vpro() {
            c_bricks(cfg, 2, I_BRICK, Some(V_BRICK))
        } else {
            c_bricks(cfg, 1, AttachedBrick { graphics_bay: Some(G_BRICK_BAY), ..I_BRICK }, None)
        }
    }

    fn wire(nodes: &[&Node], cfg: &MachineConfig) -> Result<BoardIo, String> {
        if cfg.graphics_board().is_vpro() {
            let (mut io, _) = wire_ibrick(nodes, cfg, None)?;
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
        let (mut io, xbridge) = wire_ibrick(nodes, cfg, Some(port_reset))?;
        let status = new_status();
        let pipe = fit_pipe(&nodes[0].bedrock, &xbridge, cfg, KonaConfig::new(product, rm_count)?, &status)?;
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
}

/// A V-brick on `node`'s XIO connector: an XBridge with no PCI devices, the
/// V12 at [`V12_WIDGET`], DMA and interrupt packets through the node's
/// Bedrock as any widget's are.
///
/// Established: the V-brick holds an XBridge and cables to a C-brick of its
/// own. Inferred: it has no PCI devices; the XBridge's two empty PCI heads
/// answer as the chip's always do.
fn wire_vbrick(node: &Node, cfg: &MachineConfig) -> Result<Graphics, String> {
    let bedrock = &node.bedrock;
    let xbridge = XBridge::new(BEDROCK_PORT, &|widget| {
        let port = Inbound::port(bedrock, widget);
        (port.clone() as Arc<dyn BusDevice>, port as Arc<dyn InterruptController>)
    });
    for widget in [0, HOST_WIDGET, HEAD_E, HEAD_F] {
        bedrock.connect_widget(widget, xbridge.widget(widget).expect("the V-brick's ports are the XBridge's"));
    }
    let status = new_status();
    let port = Inbound::port(bedrock, V12_WIDGET);
    let packets: PacketSender = Arc::new(move |widget, address, vector, level| {
        widget == BEDROCK_PORT.widget && port.packet(V12_WIDGET, address, vector.into(), level)
    });
    let board = cfg.graphics_board();
    let v12 = Odyssey::new_widget_relative(
        V12_WIDGET,
        board.as_str(),
        packets,
        bedrock.widget_dma(V12_WIDGET),
        status.clone(),
        (cfg.graphics.threads, cfg.graphics.tile),
        (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str()),
    )?;
    bedrock.connect_widget(V12_WIDGET, xbridge.attach_widget(V12_WIDGET, v12.clone()));
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

/// An InfiniteReality pipe at widget 11 of the I-brick's XBridge: its DMA
/// enters Bedrock as that widget's inbound traffic and its interrupt packets
/// go to the address the driver programs. Its destination widget out of
/// reset is the XBridge's host port.
fn fit_pipe(
    bedrock: &Arc<Bedrock>,
    xbridge: &Arc<XBridge>,
    cfg: &MachineConfig,
    kona: KonaConfig,
    status: &Arc<crate::disp::StatusSource>,
) -> Result<Arc<Kona>, String> {
    let kona = kona.with_host_widget(IBRICK.host_port)?.with_widget(PIPE_WIDGET)?.with_pool(cfg.graphics.threads, cfg.graphics.tile);
    let flash = cfg.graphics.flash.as_deref().map(|p| crate::dev::kona::flash_file(Path::new(p))).transpose()?;
    // Its interrupt packets are the chip's inbound writes from that widget, to
    // the address the driver programmed.
    let pipe = Kona::new(kona, Inbound::port(bedrock, PIPE_WIDGET), bedrock.widget_dma(PIPE_WIDGET), status.clone(), flash)?;
    bedrock.connect_widget(PIPE_WIDGET, xbridge.attach_widget(PIPE_WIDGET, pipe.clone()));
    Ok(pipe)
}

#[cfg(test)]
mod tests;
