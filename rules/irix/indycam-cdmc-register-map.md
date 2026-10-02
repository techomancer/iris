# IndyCam CDMC register map and power-on defaults

**Keywords:** indycam, cdmc, vino, camera, green, colour cast, white balance,
register map, i2c 0x56
**Category:** vino, irix
**Status:** Fixed in `src/dev/cdmc.rs`.

## Symptom

Live camera capture (`[vino] source = "camera"`) produced a recognisable but
uniformly **green** image — luma and geometry correct, red and blue crushed to
near zero. Reproduced on IRIX 6.5.22 with `vidtomem -f /tmp/cap -v 0`.

## Cause

Two compounding bugs in the emulated CDMC (the IndyCam's I2C controller at
slave address 0x56/0x57):

1. **The register map was wrong.** Every subaddress except VERSION (0x0E) was
   assigned to the wrong control. The real map — same silicon the Linux
   `indycam` driver drives — is:

   | Sub | Register | Access | Default |
   |-----|----------|--------|---------|
   | 0x00 | CONTROL (AGCENA, AWBCTL, EVNFLD) | rw | AGCENA |
   | 0x01 | SHUTTER | rw | 0xFF |
   | 0x02 | GAIN | rw | 0x80 |
   | 0x03 | BRIGHTNESS | r | 0x80 |
   | 0x04 | RED_BALANCE | rw | 0x18 |
   | 0x05 | BLUE_BALANCE | rw | 0xA4 |
   | 0x06 | RED_SATURATION | rw | 0x80 |
   | 0x07 | BLUE_SATURATION | rw | 0xC0 |
   | 0x08 | GAMMA | rw | 0x80 |
   | 0x0E | VERSION | r | 0x10 (IndyCam v1.0) |
   | 0x0F | RESET | w | — |

2. **The register file powered up at zero while the pixel maths treated 0x80 as
   neutral.** Every unprogrammed control therefore sat at its extreme. The
   balance and saturation terms pulled Cb and Cr from 128 down to 96, which
   through BT.601 is R−51, G+38, B−64 per pixel — exactly the observed green
   cast. Note the defaults are *not* all 0x80: red balance is 0x18 and blue
   saturation 0xC0, so "0x80 is neutral" is wrong even with a correct map.

## Fix

`apply_uyvy_field` is now anchored at the power-on values: with the defaults in
place every term is unity and the field passes through byte-identical. Controls
respond proportionally as the guest moves them away from default. AGCENA and
AWBCTL suppress the gain and balance terms respectively, which matches a host
webcam running its own auto-exposure and auto-white-balance.

`power_on_defaults_pass_the_field_through_untouched` in `src/dev/cdmc.rs` locks the
pass-through property in.

## Debugging technique worth reusing

Isolate host capture from the guest before touching emulator code. A short
example binary that opens the camera through the same `nokhwa` path, prints the
negotiated format and raw buffer length, and writes a PPM proved in one run that
macOS AVFoundation delivers correct 1920×1080 packed YUYV (Cb at byte 1, Cr at
byte 3) — clearing `src/dev/camera.rs` and pointing at the guest side.

Then replay a captured host frame through `Cdmc::apply_uyvy_field` offline with
candidate register states and compare against the guest's screenshot. The
"gain programmed, everything else zero" state reproduced the reported image
exactly, which pinned the fault to the CDMC stage without a single boot.

## See also

- [indycam-end-to-end-capture.md](indycam-end-to-end-capture.md)
- [vino-capture-on-6.5-progress.md](vino-capture-on-6.5-progress.md)
- Linux `drivers/media/video/indycam.h` — authoritative register map/defaults
