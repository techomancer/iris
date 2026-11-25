use super::hq2::Hq2Engine;
use super::re3::{Re3Context, FB_H, FB_W, REGMASK, STREAM_WRITE};
use super::{Gr2, Gr2Regs, Gr2Variant, GEDMA_OUT_WORDS};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::atomic::Ordering;

const GR2_SNAPSHOT_VERSION: u32 = 1;
const GR2_STATE_MAX_BYTES: usize = 8 << 20;

pub type Gr2SnapshotParts = (String, Vec<(String, Vec<u32>)>);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gr2Metadata {
    version: u32,
    variant: String,
    width: u32,
    height: u32,
    state_bytes: u64,
}

#[derive(Serialize, Deserialize)]
struct Gr2State {
    version: u32,
    regs: Vec<u8>,
    hq: Vec<u8>,
    re3: Vec<u8>,
    fin2_wait: u32,
    fin3_pending: u32,
    gedma: Vec<u32>,
}

fn pack_bytes(bytes: &[u8]) -> Result<Vec<u32>, String> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| format!("GR2 snapshot part is too large: {} bytes", bytes.len()))?;
    let mut words = Vec::with_capacity(1 + bytes.len().div_ceil(4));
    words.push(len);
    for chunk in bytes.chunks(4) {
        let mut word = [0; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_be_bytes(word));
    }
    Ok(words)
}

fn unpack_bytes(name: &str, words: &[u32]) -> Result<Vec<u8>, String> {
    let Some((&len, body)) = words.split_first() else {
        return Err(format!("GR2 snapshot part {name} is empty"));
    };
    let len = len as usize;
    if body.len() != len.div_ceil(4) {
        return Err(format!(
            "GR2 snapshot part {name} has {} payload words for {len} bytes",
            body.len()
        ));
    }
    let mut bytes = Vec::with_capacity(body.len() * 4);
    for word in body {
        bytes.extend_from_slice(&word.to_be_bytes());
    }
    bytes.truncate(len);
    Ok(bytes)
}

fn decode_state(bytes: &[u8]) -> Result<Gr2State, String> {
    if bytes.len() > GR2_STATE_MAX_BYTES {
        return Err(format!(
            "GR2 state is {} bytes, maximum is {GR2_STATE_MAX_BYTES}",
            bytes.len()
        ));
    }
    let (state, rest) =
        postcard::take_from_bytes(bytes).map_err(|e| format!("decode GR2 state: {e}"))?;
    if !rest.is_empty() {
        return Err(format!("GR2 state has {} trailing bytes", rest.len()));
    }
    Ok(state)
}

fn raw_bytes<T>(value: &T) -> Vec<u8> {
    let mut bytes = vec![0; size_of::<T>()];
    // SAFETY: every captured GR2 type is fixed-layout plain data. Workers
    // are stopped, so no field changes while its object representation is
    // copied.
    unsafe {
        std::ptr::copy_nonoverlapping(
            value as *const T as *const u8,
            bytes.as_mut_ptr(),
            bytes.len(),
        );
    }
    bytes
}

fn decode_raw<T>(name: &str, bytes: &[u8]) -> Result<Box<T>, String> {
    if bytes.len() != size_of::<T>() {
        return Err(format!(
            "GR2 {name} image is {} bytes, expected {}",
            bytes.len(),
            size_of::<T>()
        ));
    }
    let mut value = Box::<T>::new_uninit();
    // SAFETY: callers use this only for GR2's fixed-layout, numeric,
    // all-bit-pattern-valid plain-data structures.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), value.as_mut_ptr() as *mut u8, bytes.len());
        Ok(value.assume_init())
    }
}

fn variant_name(variant: Gr2Variant) -> &'static str {
    match variant {
        Gr2Variant::Xz => "xz",
        Gr2Variant::Elan => "elan",
        Gr2Variant::Extreme => "extreme",
    }
}

fn validate_regs(regs: &Gr2Regs, variant: Gr2Variant) -> Result<(), String> {
    if regs.ge.count != variant.ges() {
        return Err(format!(
            "GR2 register image has {} GE7s, {} requires {}",
            regs.ge.count,
            variant.name(),
            variant.ges()
        ));
    }
    for (index, xmap) in regs.xmap.iter().enumerate() {
        if xmap.addr as usize >= super::xmap5::CLUT_SIZE {
            return Err(format!("GR2 XMAP{index} address {:#x} is invalid", xmap.addr));
        }
        if xmap.clut_phase > 2 {
            return Err(format!(
                "GR2 XMAP{index} CLUT phase {} is invalid",
                xmap.clut_phase
            ));
        }
    }
    Ok(())
}

fn validate_re3(ctx: &Re3Context) -> Result<(), String> {
    for (index, (&value, &mask)) in ctx.reg.iter().zip(REGMASK.iter()).enumerate() {
        if value & !mask != 0 {
            return Err(format!(
                "GR2 RE3 register {index:#x} contains bits outside {mask:#010x}"
            ));
        }
    }
    if ctx.stream > STREAM_WRITE {
        return Err(format!("GR2 RE3 stream state {} is invalid", ctx.stream));
    }
    if ctx.pixfmt > super::re3::PIXFMT_RGB4 {
        return Err(format!("GR2 RE3 pixel format {} is invalid", ctx.pixfmt));
    }
    Ok(())
}

impl Gr2 {
    /// Complete architectural GR2 state split into exact named word streams.
    /// HQ2, RE3, and the display worker must already be stopped.
    pub fn snapshot_parts(&self) -> Result<Gr2SnapshotParts, String> {
        if self.running.load(Ordering::Acquire) {
            return Err("GR2 workers must be stopped before taking a snapshot".into());
        }
        if !self.hq_fifo.is_empty()
            || !self.re3_fifo.is_empty()
            || self.hq_busy.load(Ordering::Acquire)
            || self.re3_busy.load(Ordering::Acquire)
        {
            return Err("GR2 pipeline is not quiescent".into());
        }
        let fin3_pending = self.fin3_pending.load(Ordering::Acquire);
        if fin3_pending != 0 {
            return Err(format!(
                "GR2 has {fin3_pending} pending FIN3 command(s) at a quiescent boundary"
            ));
        }
        let head = self.gedma_out_head.load(Ordering::Acquire);
        let tail = self.gedma_out_tail.load(Ordering::Acquire);
        let gedma_len = head.wrapping_sub(tail) as usize;
        if gedma_len > GEDMA_OUT_WORDS {
            return Err(format!(
                "GR2 GEDMA output contains {gedma_len} words, capacity is {GEDMA_OUT_WORDS}"
            ));
        }
        let gedma = (0..gedma_len)
            .map(|i| {
                self.gedma_out[(tail as usize + i) & (GEDMA_OUT_WORDS - 1)]
                    .load(Ordering::Relaxed)
            })
            .collect();
        // SAFETY: the ownership contract above leaves all three objects
        // immutable until this method returns.
        let (regs, mut hq, re3) = unsafe {
            (
                raw_bytes(&*self.regs.get()),
                raw_bytes(&*self.hq_engine.get()),
                &*self.re3.get(),
            )
        };
        Hq2Engine::scrub_snapshot_image(&mut hq);
        let state = Gr2State {
            version: GR2_SNAPSHOT_VERSION,
            regs,
            hq,
            re3: raw_bytes(&re3.ctx),
            fin2_wait: self.fin2_wait.load(Ordering::Acquire),
            fin3_pending,
            gedma,
        };
        let state = postcard::to_allocvec(&state).map_err(|e| format!("encode GR2 state: {e}"))?;
        let metadata = toml::to_string(&Gr2Metadata {
            version: GR2_SNAPSHOT_VERSION,
            variant: variant_name(self.variant).into(),
            width: FB_W as u32,
            height: FB_H as u32,
            state_bytes: state.len() as u64,
        })
        .map_err(|e| format!("encode GR2 snapshot metadata: {e}"))?;
        Ok((
            metadata,
            vec![
                ("state".into(), pack_bytes(&state)?),
                ("vram".into(), re3.vram.to_vec()),
                ("zbuf".into(), re3.zbuf.to_vec()),
            ],
        ))
    }

    /// Restore a complete GR2 snapshot. All payloads are decoded and
    /// validated before any live architectural state changes.
    pub fn restore_snapshot_parts(
        &self,
        metadata: &str,
        parts: &[(String, Vec<u32>)],
    ) -> Result<(), String> {
        if self.running.load(Ordering::Acquire) {
            return Err("GR2 workers must be stopped before restoring a snapshot".into());
        }
        let metadata: Gr2Metadata =
            toml::from_str(metadata).map_err(|e| format!("decode GR2 snapshot metadata: {e}"))?;
        if metadata.version != GR2_SNAPSHOT_VERSION {
            return Err(format!(
                "GR2 snapshot version {} is not supported",
                metadata.version
            ));
        }
        if metadata.variant != variant_name(self.variant) {
            return Err(format!(
                "GR2 snapshot is for {}, running board is {}",
                metadata.variant,
                variant_name(self.variant)
            ));
        }
        if metadata.width != FB_W as u32 || metadata.height != FB_H as u32 {
            return Err(format!(
                "GR2 snapshot framebuffer is {}x{}, expected {FB_W}x{FB_H}",
                metadata.width, metadata.height
            ));
        }
        let mut named = HashMap::with_capacity(parts.len());
        for (name, words) in parts {
            if named.insert(name.as_str(), words.as_slice()).is_some() {
                return Err(format!("duplicate GR2 snapshot part {name}"));
            }
        }
        if named.len() != 3 {
            return Err(format!("GR2 snapshot has {} parts, expected 3", named.len()));
        }
        let state_words = named
            .remove("state")
            .ok_or("GR2 snapshot part state is missing")?;
        let vram = named
            .remove("vram")
            .ok_or("GR2 snapshot part vram is missing")?;
        let zbuf = named
            .remove("zbuf")
            .ok_or("GR2 snapshot part zbuf is missing")?;
        if let Some(name) = named.keys().next() {
            return Err(format!("unknown GR2 snapshot part {name}"));
        }
        if vram.len() != FB_W * FB_H {
            return Err(format!(
                "GR2 VRAM part has {} words, expected {}",
                vram.len(),
                FB_W * FB_H
            ));
        }
        if zbuf.len() != FB_W * FB_H {
            return Err(format!(
                "GR2 Z-buffer part has {} words, expected {}",
                zbuf.len(),
                FB_W * FB_H
            ));
        }
        let state_bytes = unpack_bytes("state", state_words)?;
        if metadata.state_bytes != state_bytes.len() as u64 {
            return Err(format!(
                "GR2 metadata records {} state bytes, part has {}",
                metadata.state_bytes,
                state_bytes.len()
            ));
        }
        let state = decode_state(&state_bytes)?;
        if state.version != GR2_SNAPSHOT_VERSION {
            return Err(format!("GR2 state version {} is not supported", state.version));
        }
        if state.fin3_pending != 0 {
            return Err(format!(
                "GR2 state has {} pending FIN3 command(s) but no queued FIFO entries",
                state.fin3_pending
            ));
        }
        if state.fin2_wait > 1 {
            return Err(format!("GR2 FIN2 wait flag {} is invalid", state.fin2_wait));
        }
        if state.gedma.len() > GEDMA_OUT_WORDS {
            return Err(format!(
                "GR2 GEDMA output contains {} words, capacity is {GEDMA_OUT_WORDS}",
                state.gedma.len()
            ));
        }
        let regs = decode_raw::<Gr2Regs>("register", &state.regs)?;
        let mut hq = decode_raw::<Hq2Engine>("HQ2", &state.hq)?;
        let re3 = decode_raw::<Re3Context>("RE3 context", &state.re3)?;
        validate_regs(&regs, self.variant)?;
        hq.validate_snapshot()?;
        validate_re3(&re3)?;
        hq.clear_runtime_diagnostics();

        // SAFETY: validation is complete, workers are stopped, and these are
        // plain-data objects with no destructors or runtime-owned pointers.
        unsafe {
            std::ptr::copy_nonoverlapping(&*regs, self.regs.get(), 1);
            std::ptr::copy_nonoverlapping(&*hq, self.hq_engine.get(), 1);
            let live_re3 = &mut *self.re3.get();
            live_re3.ctx = *re3;
            live_re3.vram.copy_from_slice(vram);
            live_re3.zbuf.copy_from_slice(zbuf);
            #[cfg(feature = "gr2-jit")]
            {
                let mode = live_re3.jit.mode;
                live_re3.jit = Default::default();
                live_re3.jit.mode = mode;
            }
            live_re3.reset_telemetry();
        }
        self.hq_fifo.reset();
        self.re3_fifo.reset();
        self.fin3_pending.store(0, Ordering::Release);
        self.fin2_wait.store(state.fin2_wait, Ordering::Release);
        self.fin2_armed_ns.store(0, Ordering::Release);
        self.fin2_progress.store(0, Ordering::Release);
        self.re3_cpu_writes.store(0, Ordering::Release);
        self.published_raster.reset();
        self.presentation_frames.store(0, Ordering::Release);
        self.presentation_host_read_pixels.store(0, Ordering::Release);
        self.copy_a.set(0);
        self.gedma_out_tail.store(0, Ordering::Release);
        for (slot, &word) in self.gedma_out.iter().zip(&state.gedma) {
            slot.store(word, Ordering::Relaxed);
        }
        self.gedma_out_head
            .store(state.gedma.len() as u32, Ordering::Release);
        self.head.damage();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::gr2::hq2::{PUC_COLOR, PUC_DATA, PUC_INIT, PUC_RECTI2D};
    use crate::dev::gr2::re3_tests::{live_gr2, w32};
    use crate::dev::gr2::GR2_BASE;
    use crate::traits::{BusDevice, Device, Resettable, BUS_OK};

    const FIFO_BASE: u32 = 0x40000;

    fn command(g: &Gr2, index: u32, value: u32) {
        w32(g, FIFO_BASE + index * 4, value);
    }

    fn data(g: &Gr2, value: u32) {
        command(g, PUC_DATA, value);
    }

    #[test]
    fn xz_and_extreme_round_trip_complete_state_and_partial_hq2_command() {
        for variant in [Gr2Variant::Xz, Gr2Variant::Elan, Gr2Variant::Extreme] {
            let g = live_gr2(variant);
            command(&g, PUC_INIT, 1);
            command(&g, PUC_COLOR, 7);
            command(&g, PUC_RECTI2D, 10);
            data(&g, 20);
            g.wait_idle();
            unsafe {
                (*g.re3.get()).vram[3 * FB_W + 4] = 0x0012_3456;
                (*g.re3.get()).zbuf[3 * FB_W + 4] = 0x89ab_cdef;
            }
            assert!(g.gedma_push(0xfeed_cafe));
            g.stop();
            let (metadata, parts) = g.snapshot_parts().unwrap();

            g.power_on();
            g.restore_snapshot_parts(&metadata, &parts).unwrap();
            assert_eq!(g.regs().ge.count, variant.ges());
            assert_eq!(g.vram()[3 * FB_W + 4], 0x0012_3456);
            assert_eq!(unsafe { (*g.re3.get()).zbuf[3 * FB_W + 4] }, 0x89ab_cdef);
            let readback = g.read32(GR2_BASE + 0x6a068);
            assert_eq!(readback.status, BUS_OK);
            assert_eq!(readback.data, 0xfeed_cafe);

            g.start();
            data(&g, 12);
            data(&g, 21);
            g.wait_idle();
            assert_ne!(g.vram()[20 * FB_W + 10], 0);
            g.stop();
            #[cfg(feature = "gr2-jit")]
            let jit_mode = unsafe { (*g.re3.get()).jit.mode };
            g.power_on();
            assert_eq!(g.regs().ge.count, variant.ges());
            assert_eq!(g.vram()[3 * FB_W + 4], 0);
            assert_eq!(unsafe { (*g.re3.get()).zbuf[3 * FB_W + 4] }, 0);
            assert_eq!(g.gedma_out_head.load(Ordering::Acquire), 0);
            assert_eq!(g.gedma_out_tail.load(Ordering::Acquire), 0);
            #[cfg(feature = "gr2-jit")]
            assert_eq!(unsafe { (*g.re3.get()).jit.mode }, jit_mode);
        }
    }

    #[test]
    fn restore_rejects_variant_and_named_part_mismatches() {
        let xz = live_gr2(Gr2Variant::Xz);
        xz.stop();
        let (metadata, parts) = xz.snapshot_parts().unwrap();
        let extreme = live_gr2(Gr2Variant::Extreme);
        extreme.stop();
        assert!(extreme
            .restore_snapshot_parts(&metadata, &parts)
            .unwrap_err()
            .contains("running board"));

        let mut missing = parts.clone();
        missing.pop();
        assert!(xz
            .restore_snapshot_parts(&metadata, &missing)
            .unwrap_err()
            .contains("expected 3"));
        let mut duplicate = parts.clone();
        duplicate[2].0 = "vram".into();
        assert!(xz
            .restore_snapshot_parts(&metadata, &duplicate)
            .unwrap_err()
            .contains("duplicate"));
        let mut renamed = parts;
        renamed[2].0 = "depth".into();
        assert!(xz
            .restore_snapshot_parts(&metadata, &renamed)
            .unwrap_err()
            .contains("zbuf"));
    }

    #[test]
    fn invalid_late_state_does_not_mutate_the_board() {
        let g = live_gr2(Gr2Variant::Xz);
        g.stop();
        let (metadata, mut parts) = g.snapshot_parts().unwrap();
        let state_part = parts.iter_mut().find(|(name, _)| name == "state").unwrap();
        let bytes = unpack_bytes("state", &state_part.1).unwrap();
        let mut state = decode_state(&bytes).unwrap();
        state.re3[4 * size_of::<u32>()..5 * size_of::<u32>()]
            .copy_from_slice(&2u32.to_ne_bytes());
        let bytes = postcard::to_allocvec(&state).unwrap();
        state_part.1 = pack_bytes(&bytes).unwrap();
        let mut metadata_value: Gr2Metadata = toml::from_str(&metadata).unwrap();
        metadata_value.state_bytes = bytes.len() as u64;
        let metadata = toml::to_string(&metadata_value).unwrap();

        unsafe {
            (*g.re3.get()).vram[0] = 0x00ab_cdef;
        }
        assert!(g
            .restore_snapshot_parts(&metadata, &parts)
            .unwrap_err()
            .contains("outside"));
        assert_eq!(g.vram()[0], 0x00ab_cdef);
    }

    #[test]
    fn snapshot_requires_stopped_workers() {
        let g = live_gr2(Gr2Variant::Xz);
        assert!(g.snapshot_parts().unwrap_err().contains("must be stopped"));
    }
}
