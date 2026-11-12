# Boot chimes: what each machine's PROM plays, and how to check it

**Keywords:** boot chime,startup tune,power-on tune,hal2,hpc3,rad1,mace,ice1724,ip22,ip24,ip28,ip30,ip32,IRIS_AUDIO_DUMP

iris carries no chime samples. Each PROM builds or decodes its tune in RAM with
its own code and starts the audio hardware playing it, so the sound is right
exactly when the CPU, FPU, DMA and audio models are. Nothing in iris picks or
substitutes a tune: another PROM revision plays another tune, and `prom = "<path>"`
selects one. A chime is a cheap, disk-free end-to-end check of those models.

| Machine | PROM | Tune | Rate |
|---|---|---|---|
| Indy IP24 | 070-9101-011 (embedded) | stored 4-bit ADPCM | 44.1 kHz |
| Indigo2 IP22 | 070-1367-012 (embedded) | synthesized plucked strings | 44.1 kHz |
| Indigo2 Impact IP28 | 070-1477-002 (embedded) | the same strings, R10000 build | 44.1 kHz |
| Octane IP30 | 4.17 (embedded) | stored 4-bit ADPCM | 32 kHz |
| O2 IP32 | 4.18 (embedded) | synthesized plucked strings | 34 kHz |

## Running a check

Run the machine headless with `IRIS_AUDIO_DUMP=<file>` set, wait for the tune,
and stop it:

```sh
IRIS_AUDIO_DUMP=chime.dump iris --headless --config indigo2.toml
```

The file holds what the audio card played, at the card's own rate before
resampling, with or without a host sound device: records of the rate (u32) and
the frame count (u32), then that many left, right pairs of little-endian i16.
The O2 has two outputs and puts the output number (u8) in front of each record.
The ICE1724 records the front pair of its mix. Each card records where it makes
its frames: HAL2 before its output gain, RAD1 after its mute words, MACE after
the DAC gain. (`[audio] output = { file = ... }` writes a WAV of what the host plays instead.)

What the dump asserts is the guest's work: the samples are what the PROM
computed and handed to the card, in order, at the rate it programmed. A tune
is therefore checked by comparing the dump with an independent computation of
the same tune. It does not exercise the host audio device or the resampler.

Requirements:

- The Indy and Indigo2 need audio on: `[machine] audio = false` removes HAL2, so
  nothing plays and nothing is recorded.
- The IP28 needs `banks = [512, 512, 0, 0]` to chime.
- The Octane run needs the config's SCSI image files to exist. The O2 needs a
  flash file (a new one is fine).
- A blank Indy or Indigo2 settings block is seeded with the factory defaults, so
  the first boot chimes (see [nvram-persistence.md](nvram-persistence.md)).
- The IP22 tune takes the longest to start, on the order of a minute or more on
  the interpreter.

## What the repository itself verifies

The recorder and each card's hook have unit tests that assert the exact bytes
recorded: `cargo test -p iris --lib -- audio_dump hal2 rad1 mace_audio ice1724
host_pcm`. `test/audiotest/check.py` reads the same dump (`--o2` for the O2's
output byte) to judge the audio sweep; see [audio-test-suite.md](audio-test-suite.md).
The repository holds no model of any PROM tune.

## IP35

The IP35 PROM does not touch the ICE1724 on iris's Tezro model (observed in a
headless run).

## Other PROM revisions

Set `prom` to another revision and its chime plays; an older 5.1 IP22 PROM
plays a stored ADPCM tune where the embedded 5.3 one synthesizes.
