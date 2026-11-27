# Serial peripherals

Reference material for the devices SGI machines talked to over their serial
ports. These documents describe the devices' own wire protocols, gathered
from public drivers, converter projects and manufacturer manuals. They are
the basis for connecting such devices to an emulated machine, whether a real
device is passed through from a host serial port or a host USB device is
converted to the serial protocol the guest expects.

| Document | Covers |
|---|---|
| [midi.md](midi.md) | The MIDI 1.0 byte stream, how Indy, Indigo2, O2 and the IOC3 machines clock a serial port at 31 250 bit/s for a Macintosh-style MIDI interface, and how the stream maps to host MIDI ports or a software synthesizer |
| [spaceball.md](spaceball.md) | The Spaceball 2003/3003/4000 FLX/5000 FLX serial protocol, the Magellan (SpaceMouse) protocol, and converting a USB SpaceNavigator to a serial Spaceball |
| [wacom-protocol-v.md](wacom-protocol-v.md) | Wacom serial Protocol V (Intuos, Intuos2): commands, replies, the 9-byte packets and tool IDs, with a short section on Protocol IV |

IRIX has its own drivers for several of these (Spaceball, Magellan, Wacom
tablets, dial and button boxes). Which protocol version each IRIX driver
speaks is still being confirmed; the documents mark where that matters.
