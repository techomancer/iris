# HAL2 and CS4216: speaker volume is the PBUS PIO ch2 MDAC (`0x1fbd8800`/`0x1fbd8804`), and clocking is SM3-Slave from HAL2

## Symptom

Sound volume adjustment in IRIX (`audiopanel`, `alSetParams` on `AL_GAIN` / `AL_MUTE`, and front-panel volume buttons), NetBSD (`haltwo`), and the PROM `volume` environment variable had no audible effect on Indy and Indigo2.

## Cause

1. **Two-stage output volume hardware**:
   - The Crystal CS4216 codec (`HAL2_CODECA_CTRL2_W`, `IAR = 0x1408`) has an internal `MUTE` bit (`IDR0` bit 10) and 5-bit Left/Right DAC attenuators `LA4:0` (`IDR1` bits 11:7) and `RA4:0` (`IDR1` bits 6:2, $-1.5\,\text{dB}$/step), which feed **Line Out**.
   - In IRIX (`a2_dd.o`), `LineOut` has a fixed `0.0 dB` gain range (`hal2_lineout_gain_range` has `min = 0, max = 0`) and `hal2_write_source_and_gain` always writes `0` to `MUTE`, `LA`, and `RA`.
   - Instead, **Speaker / Headphone Out** is attenuated after the CS4216 DAC by an external **8-bit Multiplying DAC (MDAC)** on HPC3 PBUS PIO Channel 2 at `0x1fbd8800` (`VOLUME_RIGHT`) and `0x1fbd8804` (`VOLUME_LEFT`).
   - IRIX `a2_dd.o` (`hal2_init`, `hal2_set_analog_out_if_params`, `hal2_volumectrl`, and `dezipper_output_atten`), the IP22/IP26/IP28 PROM startup tune, and NetBSD (`haltwo.c`) all adjust output volume by writing `0..255` to `0x1fbd8800` and `0x1fbd8804` (`20 * log10(V / 240)` dB in IRIX, linear voltage $V / 255$). Those writes landed in `Hpc3State::pbus_pio[0x200..0x201]` and never reached `Hal2`.

2. **Clocking and control bus (`SM3-Slave`, no $I^2C$)**:
   - The CS4216 codecs on Indy/Indigo2 are strapped to **Serial Mode 3 Slave (`SM3-Slave`)** (`PB/SB = 1`, `CM1:0 = 11`, `MS/SL = 0`). There is no $I^2C$ or microprocessor bus to the codecs; HAL2 embeds the 28-bit control word (`IDR0[15:0]` and `IDR1[11:0]`) into time slots 4, 7, and 8 of every 64-bit serial audio frame.
   - `CLKIN` ($256 F_s$), `SCLK` ($64 F_s$), and `SSYNC` ($F_s$) are all inputs to the CS4216 driven by HAL2's Bresenham clock generators (`BRES1`–`BRES3`) from the board's `24.576 MHz` ($512 \times 48\,\text{kHz}$) and `22.5792 MHz` ($512 \times 44.1\,\text{kHz}$) crystals.
   - In 4-channel quad mode (`ISR.CODEC_MODE = 1`), both CS4216 codecs share the same frame sync, so `a2_dd.o` (`hal2_set_channel_mode`) switches `analog_in_dev` from `BRES1` (`cg1`) to `BRES2` (`cg2`).

3. **Two-word indirect `CTRL2` readback (`_R0` vs `_R1`)**:
   - Reading two-word `CTRL2` registers (`HAL2_CODECA_CTRL2_R0/R1` at `0x1488`/`0x1489`, `HAL2_CODECB_CTRL2_R0/R1` at `0x1588`/`0x1589`, `HAL2_BRESn_CTRL2_R0/R1` at `0x2188`/`0x2189`..`0x2389`) uses IAR bit 0 to select which 16-bit word is placed into `IDR0`. Masking IAR with `0x000C` dropped bit 0 and returned word 0 for both `_R0` and `_R1`.

## Rule

- Forward HPC3 PBUS PIO channel 2 writes (`0x58800` = `VOLUME_RIGHT`, `0x58804` = `VOLUME_LEFT`, both 8-bit and 32-bit, plus snapshot restore from `pbus_pio[0x200..0x201]`) to `Hal2`.
- Combine the external MDAC gain (`vol_left / 255`, `vol_right / 255`, defaulting to `0xFF` before the guest writes the MDAC) with CS4216 Codec A `MUTE` (`IDR0` bit 10) and 5-bit `LA`/`RA` DAC attenuation (`IDR1` bits 11:7 and 6:2, $-1.5\,\text{dB}$/step) on live Codec A samples.
- Check `val & 1` on `IAR_PARAM_2` indirect reads so `0x..89` (`_R1`) returns the second 16-bit control word in `IDR0`.
