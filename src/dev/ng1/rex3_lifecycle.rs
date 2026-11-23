//! Stopped resident snapshot, restore and reset boundaries.

use std::io;
use std::path::Path;
use std::sync::atomic::Ordering;

use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

use super::{RasterMode, Rex3};
use crate::dev::ng1::wgpu::resident::{
    ObservationReason, PlaneMask, ResidentError, PLANE_WORDS,
};
use crate::traits::{BUS_BUSY, BUS_ERR};

pub(super) struct ResidentLifecycle {
    registers: Option<toml::Value>,
    planes: Option<(Vec<u32>, Vec<u32>)>,
    rejected: bool,
    timing_preset_pending: bool,
}

impl Default for ResidentLifecycle {
    fn default() -> Self {
        Self { registers: None, planes: None, rejected: false, timing_preset_pending: true }
    }
}

impl ResidentLifecycle {
    pub(super) fn ready(&self) -> bool {
        !self.rejected && self.registers.is_none() && self.planes.is_none()
    }

    fn reject_pending(&mut self) {
        if !self.ready() {
            self.registers = None;
            self.planes = None;
            self.rejected = true;
        }
    }

    fn resume_after_rejection(&mut self) {
        if self.rejected {
            let timing_preset_pending = self.timing_preset_pending;
            *self = Self::default();
            self.timing_preset_pending = timing_preset_pending;
        }
    }
}

impl Rex3 {
    /// Bus operations retain this guard through all producer side effects,
    /// including the return of a bulk HOSTR transfer.
    pub(super) fn resident_bus_guard(
        &self,
    ) -> Result<Option<RwLockReadGuard<'_, ResidentLifecycle>>, u32> {
        if !self.resident_frame.is_active() {
            return Ok(None);
        }
        let guard = self.resident_lifecycle.try_read().ok_or(BUS_BUSY)?;
        if !self.running.load(Ordering::Acquire)
            || self.resident_fatal_notified.load(Ordering::Acquire)
        {
            return Err(BUS_ERR);
        }
        Ok(Some(guard))
    }

    pub(super) fn resident_monitor_guard(
        &self,
    ) -> Result<Option<RwLockWriteGuard<'_, ResidentLifecycle>>, String> {
        if !self.resident_frame.is_active() {
            return Ok(None);
        }
        let guard = self.resident_lifecycle.try_write()
            .ok_or_else(|| "REX3 resident lifecycle is busy".to_owned())?;
        if !guard.ready() {
            return Err("resident restore is incomplete".into());
        }
        self.check_resident_health(false).map_err(|error| error.to_string())?;
        if self.running.load(Ordering::Acquire) {
            self.wait_resident_idle().map_err(|error| error.to_string())?;
        } else {
            self.check_resident_quiescent().map_err(|error| error.to_string())?;
        }
        Ok(Some(guard))
    }

    pub(super) fn check_resident_health(&self, reset: bool) -> Result<(), ResidentError> {
        let result = {
            let mut mode = self.raster.lock();
            match &mut *mode {
                RasterMode::Resident(raster) => match raster.fault() {
                    Some(ResidentError::DeviceLost(_) | ResidentError::Failed(_)) => {
                        Err(raster.fault().unwrap().clone())
                    }
                    Some(_) if reset => Ok(()),
                    Some(error) => Err(error.clone()),
                    None => raster.poll_device_health(),
                },
                _ => Err(ResidentError::InvalidState("resident raster is not active")),
            }
        };
        if let Err(error) = &result {
            if !matches!(error, ResidentError::InvalidState("resident raster is not active")) {
                self.fail_resident(error.clone());
            }
        }
        result
    }

    fn resident_boundary(&self, reset: bool) -> Result<(), ResidentError> {
        self.check_resident_health(reset)?;
        self.check_resident_quiescent()
    }

    fn invalidate_resident_capture(&self) {
        // A consumer checks availability again while holding this same lock.
        let mut screen = self.head.screen().lock();
        screen.width = 0;
        screen.height = 0;
    }

    pub(crate) fn apply_resident_display_resolution(
        &self,
        mode: crate::dev::ng1::vc2_timings::NewportResolution,
        initial_only: bool,
    ) -> Result<(), ResidentError> {
        let mut lifecycle = self.resident_lifecycle.write();
        self.check_resident_health(false)?;
        if !lifecycle.ready() {
            return Err(ResidentError::InvalidState("resident restore is incomplete"));
        }
        if (initial_only && !lifecycle.timing_preset_pending) || mode.is_guest() {
            return Ok(());
        }
        self.check_resident_quiescent()?;
        crate::dev::ng1::vc2_timings::apply_newport_resolution(&mut self.vc2.lock(), mode);
        lifecycle.timing_preset_pending = false;
        // The usual host background is a CPU-shadow fill; resident planes
        // remain authoritative and are deliberately left untouched.
        self.invalidate_resident_capture();
        Ok(())
    }

    pub(super) fn snapshot_resident_registers(&self) -> Result<toml::Value, ResidentError> {
        let lifecycle = self.resident_lifecycle.write();
        self.resident_boundary(false)?;
        if !lifecycle.ready() {
            return Err(ResidentError::InvalidState("resident restore is incomplete"));
        }
        Ok(self.save_state_fields())
    }

    pub(super) fn snapshot_resident_planes(&self) -> Result<(Vec<u32>, Vec<u32>), ResidentError> {
        let lifecycle = self.resident_lifecycle.write();
        self.observe_resident_planes(&lifecycle)
    }

    pub(crate) fn snapshot_resident_checkpoint(
        &self,
    ) -> Result<(toml::Value, (Vec<u32>, Vec<u32>)), ResidentError> {
        let lifecycle = self.resident_lifecycle.write();
        let planes = self.observe_resident_planes(&lifecycle)?;
        Ok((self.save_state_fields(), planes))
    }

    fn observe_resident_planes(&self, lifecycle: &ResidentLifecycle) -> Result<(Vec<u32>, Vec<u32>), ResidentError> {
        self.resident_boundary(false)?;
        if !lifecycle.ready() {
            return Err(ResidentError::InvalidState("resident restore is incomplete"));
        }
        let result = {
            let mut mode = self.raster.lock();
            match &mut *mode {
                RasterMode::Resident(raster) => raster.observe(ObservationReason::Snapshot, PlaneMask::Both),
                _ => Err(ResidentError::InvalidState("resident raster is not active")),
            }
        };
        let observation = match result {
            Ok(observation) => observation,
            Err(error) => {
                self.fail_resident(error.clone());
                return Err(error);
            }
        };
        match Self::decode_observed_planes(&observation, ObservationReason::Snapshot) {
            Some(planes) => Ok(planes),
            None => {
                let error = ResidentError::Failed("incomplete resident snapshot observation".into());
                self.fail_resident(error.clone());
                Err(error)
            }
        }
    }

    fn decode_observed_planes(
        observation: &crate::dev::ng1::wgpu::resident::Observation,
        reason: ObservationReason,
    ) -> Option<(Vec<u32>, Vec<u32>)> {
        let decode = |bytes: Option<&[u8]>| -> Option<Vec<u32>> {
            let bytes = bytes.filter(|bytes| bytes.len() == PLANE_WORDS * size_of::<u32>())?;
            Some(bytes.chunks_exact(4).map(|word| {
                u32::from_ne_bytes(word.try_into().expect("four-byte framebuffer word"))
            }).collect())
        };
        if observation.reason() == reason && observation.planes() == PlaneMask::Both {
            decode(observation.rgb()).zip(decode(observation.aux()))
        } else {
            None
        }
    }

    /// Both planes as the GPU holds them, with the board running but its
    /// GFIFO drained: a recording's checkpoint on the GPU raster. Unlike a
    /// snapshot it does not need the worker stopped.
    pub(crate) fn observe_resident_checkpoint(&self) -> Result<(Vec<u32>, Vec<u32>), ResidentError> {
        self.wait_resident_idle()?;
        let observation = {
            let mut mode = self.raster.lock();
            match &mut *mode {
                RasterMode::Resident(raster) => raster.observe(ObservationReason::Recording, PlaneMask::Both)?,
                _ => return Err(ResidentError::InvalidState("resident raster is not active")),
            }
        };
        Self::decode_observed_planes(&observation, ObservationReason::Recording)
            .ok_or_else(|| ResidentError::Failed("incomplete resident checkpoint observation".into()))
    }

    pub(super) fn save_resident_framebuffers(&self, dir: &Path, prefix: &str) -> io::Result<()> {
        let lifecycle = self.resident_lifecycle.write();
        let (rgb, aux) = self.observe_resident_planes(&lifecycle).map_err(io::Error::other)?;
        Self::write_captured_resident_framebuffers(dir, prefix, &rgb, &aux)
    }

    pub(crate) fn write_captured_resident_framebuffers(
        dir: &Path,
        prefix: &str,
        rgb: &[u32],
        aux: &[u32],
    ) -> io::Result<()> {
        validate_planes(rgb.len(), aux.len()).map_err(io::Error::other)?;
        let pending = incomplete_framebuffer_save(dir, prefix);
        std::fs::write(&pending, b"REX3 resident framebuffer save is incomplete\n")?;
        Self::write_framebuffer_files(dir, prefix, rgb, aux)?;
        std::fs::remove_file(pending)
    }

    pub(super) fn stage_resident_registers(&self, state: &toml::Value) -> Result<(), String> {
        let mut lifecycle = self.resident_lifecycle.write();
        self.resident_boundary(false).map_err(|error| error.to_string())?;
        if let Err(error) = validate_state(&self.save_state_fields(), state, "REX3")
            .and_then(|()| validate_continuation(state))
        {
            lifecycle.reject_pending();
            return Err(error);
        }
        if lifecycle.registers.is_some() {
            lifecycle.reject_pending();
            return Err("resident register restore is already staged".into());
        }
        lifecycle.resume_after_rejection();
        lifecycle.registers = Some(state.clone());
        self.invalidate_resident_capture();
        self.commit_resident_restore(&mut lifecycle).map_err(|error| error.to_string())
    }

    pub(super) fn stage_resident_planes(&self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        let mut lifecycle = self.resident_lifecycle.write();
        self.resident_boundary(false)?;
        if let Err(error) = validate_planes(rgb.len(), aux.len()) {
            lifecycle.reject_pending();
            return Err(error);
        }
        if lifecycle.planes.is_some() {
            lifecycle.reject_pending();
            return Err(ResidentError::InvalidState("resident plane restore is already staged"));
        }
        lifecycle.resume_after_rejection();
        lifecycle.planes = Some((rgb.to_vec(), aux.to_vec()));
        self.invalidate_resident_capture();
        self.commit_resident_restore(&mut lifecycle)
    }

    pub(super) fn load_resident_framebuffers(&self, dir: &Path, prefix: &str) -> io::Result<()> {
        let mut lifecycle = self.resident_lifecycle.write();
        self.resident_boundary(false).map_err(io::Error::other)?;
        let read = |plane: &str| -> io::Result<Vec<u32>> {
            let bytes = std::fs::read(dir.join(format!("{prefix}_{plane}.bin")))?;
            if bytes.len() != PLANE_WORDS * size_of::<u32>() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("incomplete resident {plane} plane")));
            }
            Ok(bytes.chunks_exact(4).map(|word| {
                u32::from_be_bytes(word.try_into().expect("four-byte framebuffer word"))
            }).collect())
        };
        let planes = (|| {
            if incomplete_framebuffer_save(dir, prefix).try_exists()? {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "resident framebuffer save is incomplete"));
            }
            Ok((read("rgb")?, read("aux")?))
        })();
        let (rgb, aux) = match planes {
            Ok(planes) => planes,
            Err(error) => {
                lifecycle.reject_pending();
                return Err(error);
            }
        };
        if lifecycle.planes.is_some() {
            lifecycle.reject_pending();
            return Err(io::Error::other("resident plane restore is already staged"));
        }
        lifecycle.resume_after_rejection();
        lifecycle.planes = Some((rgb, aux));
        self.invalidate_resident_capture();
        self.commit_resident_restore(&mut lifecycle).map_err(io::Error::other)
    }

    fn commit_resident_restore(&self, lifecycle: &mut ResidentLifecycle) -> Result<(), ResidentError> {
        let (Some(state), Some((rgb, aux))) = (&lifecycle.registers, &lifecycle.planes) else {
            return Ok(());
        };
        let result = {
            let mut mode = self.raster.lock();
            match &mut *mode {
                RasterMode::Resident(raster) => raster.restore(rgb, aux),
                _ => Err(ResidentError::InvalidState("resident raster is not active")),
            }
        };
        if let Err(error) = result {
            lifecycle.reject_pending();
            self.fail_resident(error.clone());
            return Err(error);
        }
        self.load_state_fields(state);
        self.xmap_fence.store(0, Ordering::Release);
        self.gfifo_fence.store(0, Ordering::Release);
        *lifecycle = ResidentLifecycle::default();
        lifecycle.timing_preset_pending = false;
        self.invalidate_resident_capture();
        Ok(())
    }

    pub(super) fn reset_resident_board(&self) -> Result<(), ResidentError> {
        let mut lifecycle = self.resident_lifecycle.write();
        self.resident_boundary(true)?;
        self.invalidate_resident_capture();
        let zeros = vec![0u32; PLANE_WORDS];
        let result = {
            let mut mode = self.raster.lock();
            match &mut *mode {
                RasterMode::Resident(raster) => raster.reset(&zeros, &zeros),
                _ => Err(ResidentError::InvalidState("resident raster is not active")),
            }
        };
        if let Err(error) = result {
            self.fail_resident(error.clone());
            return Err(error);
        }
        self.reset_registers();
        self.gfifo.reset();
        self.gfxbusy.store(false, Ordering::Release);
        self.xmap_fence.store(0, Ordering::Release);
        self.gfifo_fence.store(0, Ordering::Release);
        *lifecycle = ResidentLifecycle::default();
        self.resident_fatal_notified.store(false, Ordering::Release);
        Ok(())
    }
}

fn incomplete_framebuffer_save(dir: &Path, prefix: &str) -> std::path::PathBuf {
    dir.join(format!("{prefix}_resident_incomplete"))
}

fn validate_planes(rgb: usize, aux: usize) -> Result<(), ResidentError> {
    for (plane, actual) in [("RGB", rgb), ("auxiliary", aux)] {
        if actual != PLANE_WORDS {
            return Err(ResidentError::InvalidSeed { plane, expected: PLANE_WORDS, actual });
        }
    }
    Ok(())
}

fn validate_state(template: &toml::Value, state: &toml::Value, path: &str) -> Result<(), String> {
    match (template, state) {
        (toml::Value::Table(expected), toml::Value::Table(actual)) => {
            for (key, value) in expected {
                let name = format!("{path}.{key}");
                let entry = actual.get(key).ok_or_else(|| format!("missing {name}"))?;
                validate_state(value, entry, &name)?;
            }
            Ok(())
        }
        (toml::Value::Array(expected), toml::Value::Array(actual)) if expected.len() == actual.len() => {
            for (value, entry) in expected.iter().zip(actual) {
                validate_state(value, entry, path)?;
            }
            Ok(())
        }
        (toml::Value::String(example), value) => {
            let bits = example.len().saturating_sub(2) * 4;
            let max = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
            if crate::snapshot::toml_u64(value).is_some_and(|number| number <= max) {
                Ok(())
            } else {
                Err(format!("invalid {path}"))
            }
        }
        _ => Err(format!("incomplete or invalid {path}")),
    }
}

fn validate_continuation(state: &toml::Value) -> Result<(), String> {
    for (section, key, max) in [
        ("context", "hostcnt", 8),
        ("context", "pat_bit", 31),
        ("context", "zpat_bit", 31),
        ("context", "mid_primitive", 1),
        ("cmap0", "rgb_counter", 2),
        ("cmap1", "rgb_counter", 2),
        ("bt445", "rgb_counter", 2),
    ] {
        let number = state.get(section).and_then(|v| v.get(key)).and_then(crate::snapshot::toml_u64);
        if !number.is_some_and(|number| number <= max) {
            return Err(format!("invalid REX3.{section}.{key}"));
        }
    }
    for (section, keys) in [
        ("vc2", &["index"][..]),
        ("xmap0", &["config", "cursor_cmap_msb", "popup_cmap_msb", "mode_addr", "fifo_avail"][..]),
        ("xmap1", &["config", "cursor_cmap_msb", "popup_cmap_msb", "mode_addr", "fifo_avail"][..]),
        ("cmap0", &["addr_lo", "addr_hi", "rgb_counter", "red_temp", "green_temp", "command", "revision"][..]),
        ("cmap1", &["addr_lo", "addr_hi", "rgb_counter", "red_temp", "green_temp", "command", "revision"][..]),
    ] {
        for key in keys {
            let value = state.get(section).and_then(|v| v.get(*key)).and_then(crate::snapshot::toml_u64);
            if !value.is_some_and(|number| number <= u8::MAX as u64) {
                return Err(format!("invalid REX3.{section}.{key}"));
            }
        }
    }
    let regs = state.get("vc2").and_then(|v| v.get("regs")).and_then(toml::Value::as_array);
    if !regs.is_some_and(|regs| regs.iter().all(|value| {
        crate::snapshot::toml_u64(value).is_some_and(|number| number <= u16::MAX as u64)
    })) {
        return Err("invalid REX3.vc2.regs".into());
    }
    Ok(())
}
