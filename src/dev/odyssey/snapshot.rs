use super::board::BoardSnapshot;
#[cfg(test)]
use super::board::{ODSY_HEIGHT, ODSY_WIDTH};
use super::display_fifo::DisplayFifoSnapshot;
use super::interface::HostInterfaceSnapshot;
use super::reader::DisplayRegsSnapshot;
use super::Odyssey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

const ODYSSEY_SNAPSHOT_VERSION: u32 = 1;
const ODYSSEY_STATE_MAX_BYTES: usize = 16 << 20;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OdysseyMetadata {
    version: u32,
    board: String,
    memory_bytes: u64,
}

#[derive(Serialize, Deserialize)]
struct OdysseyState {
    version: u32,
    board: BoardSnapshot,
    front: HostInterfaceSnapshot,
    display_fifo: DisplayFifoSnapshot,
    display_regs: DisplayRegsSnapshot,
    line_phase: u32,
}

fn pack_bytes(bytes: &[u8]) -> Result<Vec<u32>, String> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| format!("VPro snapshot part is too large: {} bytes", bytes.len()))?;
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
        return Err(format!("VPro snapshot part {name} is empty"));
    };
    let len = len as usize;
    if body.len() != len.div_ceil(4) {
        return Err(format!(
            "VPro snapshot part {name} has {} payload words for {len} bytes",
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

fn pack_vram(words: &[u64], bytes: u64) -> Result<Vec<u32>, String> {
    let len = u32::try_from(bytes).map_err(|_| format!("VPro VRAM is too large: {bytes} bytes"))?;
    let mut packed = Vec::with_capacity(1 + words.len() * 2);
    packed.push(len);
    for word in words {
        let bytes = word.to_ne_bytes();
        packed.push(u32::from_be_bytes(
            bytes[..4].try_into().expect("four-byte halfword"),
        ));
        packed.push(u32::from_be_bytes(
            bytes[4..].try_into().expect("four-byte halfword"),
        ));
    }
    Ok(packed)
}

fn decode_state(bytes: &[u8]) -> Result<OdysseyState, String> {
    if bytes.len() > ODYSSEY_STATE_MAX_BYTES {
        return Err(format!(
            "VPro state is {} bytes, maximum is {ODYSSEY_STATE_MAX_BYTES}",
            bytes.len()
        ));
    }
    let (state, rest) =
        postcard::take_from_bytes(bytes).map_err(|e| format!("decode VPro state: {e}"))?;
    if !rest.is_empty() {
        return Err(format!("VPro state has {} trailing bytes", rest.len()));
    }
    Ok(state)
}

impl Odyssey {
    /// Architectural state and VRAM split into named word streams. The
    /// board and raster workers must already be stopped.
    pub fn snapshot_parts(&self) -> Result<(String, Vec<(String, Vec<u32>)>), String> {
        if self.running.load(Ordering::Acquire) {
            return Err("VPro workers must be stopped before taking a snapshot".into());
        }
        if self.state.fifo.level() != 0 {
            return Err("VPro command FIFO is not empty".into());
        }
        if let Some(error) = self.state.shared.gpu_fault.lock().as_ref() {
            return Err(format!("VPro GPU raster unit is faulted: {error}"));
        }
        self.state.with_stopped_board(|board| {
            board.gpu_sync_result(super::gpu_unit::Observation::Snapshot)
        })?;
        if let Some(error) = self.state.shared.gpu_fault.lock().as_ref() {
            return Err(format!("VPro GPU raster unit is faulted: {error}"));
        }
        let (board_name, memory_bytes, board, vram, line_phase) =
            self.state.with_stopped_board(|board| {
                let memory_bytes = board.vram.size();
                let memory = board.vram.ptr_shared();
                let vram = pack_vram(memory.words_at(0, memory_bytes), memory_bytes)?;
                Ok::<_, String>((
                    board.widget.product.clone(),
                    memory_bytes,
                    BoardSnapshot::capture(board)?,
                    vram,
                    board.snapshot_line_phase()?,
                ))
            })?;
        let state = OdysseyState {
            version: ODYSSEY_SNAPSHOT_VERSION,
            board,
            front: self.state.front.lock().snapshot(),
            display_fifo: self.state.display_fifo.lock().snapshot(),
            display_regs: DisplayRegsSnapshot::capture(&self.state.shared.regs),
            line_phase,
        };
        let state = postcard::to_allocvec(&state).map_err(|e| format!("encode VPro state: {e}"))?;
        let metadata = toml::to_string(&OdysseyMetadata {
            version: ODYSSEY_SNAPSHOT_VERSION,
            board: board_name,
            memory_bytes,
        })
        .map_err(|e| format!("encode VPro snapshot metadata: {e}"))?;
        Ok((
            metadata,
            vec![("state".into(), pack_bytes(&state)?), ("vram".into(), vram)],
        ))
    }

    /// Restore a complete VPro snapshot after the machine's power-on reset.
    /// Every named payload is decoded and validated before live state changes.
    pub fn restore_snapshot_parts(
        &self,
        metadata: &str,
        parts: &[(String, Vec<u32>)],
    ) -> Result<(), String> {
        if self.running.load(Ordering::Acquire) {
            return Err("VPro workers must be stopped before restoring a snapshot".into());
        }
        let metadata: OdysseyMetadata =
            toml::from_str(metadata).map_err(|e| format!("decode VPro snapshot metadata: {e}"))?;
        if metadata.version != ODYSSEY_SNAPSHOT_VERSION {
            return Err(format!(
                "VPro snapshot version {} is not supported",
                metadata.version
            ));
        }
        let mut named = HashMap::with_capacity(parts.len());
        for (name, words) in parts {
            if named.insert(name.as_str(), words.as_slice()).is_some() {
                return Err(format!("duplicate VPro snapshot part {name}"));
            }
        }
        if named.len() != 2 {
            return Err(format!(
                "VPro snapshot has {} parts, expected 2",
                named.len()
            ));
        }
        let state_words = named
            .remove("state")
            .ok_or("VPro snapshot part state is missing")?;
        let vram_words = named
            .remove("vram")
            .ok_or("VPro snapshot part vram is missing")?;
        if let Some(name) = named.keys().next() {
            return Err(format!("unknown VPro snapshot part {name}"));
        }
        let state = decode_state(&unpack_bytes("state", state_words)?)?;
        if state.version != ODYSSEY_SNAPSHOT_VERSION {
            return Err(format!(
                "VPro state version {} is not supported",
                state.version
            ));
        }
        let vram = unpack_bytes("vram", vram_words)?;
        self.state.with_stopped_board(|board| {
            if metadata.board != board.widget.product {
                return Err(format!(
                    "VPro snapshot is for {}, running board is {}",
                    metadata.board, board.widget.product
                ));
            }
            if metadata.memory_bytes != board.vram.size() {
                return Err(format!(
                    "VPro snapshot has {} bytes of memory, running board has {}",
                    metadata.memory_bytes,
                    board.vram.size()
                ));
            }
            if vram.len() as u64 != metadata.memory_bytes {
                return Err(format!(
                    "VPro VRAM part is {} bytes, expected {}",
                    vram.len(),
                    metadata.memory_bytes
                ));
            }
            state.board.validate(board)
        })?;
        let current_identity = self.state.front.lock().identity;
        state.front.validate(current_identity)?;
        state.display_fifo.validate()?;
        state.display_regs.validate()?;

        let OdysseyState {
            board,
            front,
            display_fifo,
            display_regs,
            line_phase,
            ..
        } = state;
        self.state.with_stopped_board(|running| {
            let ptr = running.vram.ptr_shared();
            for (index, bytes) in vram.chunks(64 << 10).enumerate() {
                let mut words = Vec::with_capacity(bytes.len() / 8);
                for word in bytes.chunks_exact(8) {
                    words.push(u64::from_ne_bytes(
                        word.try_into().expect("eight-byte VRAM word"),
                    ));
                }
                ptr.write_words_untracked((index * (64 << 10)) as u64, &words);
            }
            ptr.mark_written();
            running.restore_snapshot(board)?;
            running.restore_raster_snapshot(line_phase)
        })?;
        self.state.front.lock().restore_snapshot(front)?;
        self.state
            .display_fifo
            .lock()
            .restore_snapshot(display_fifo)?;
        self.state.shared.regs.restore_snapshot(display_regs)?;
        let gpu_enabled = self.state.with_stopped_board(|board| board.gpu_unit());
        self.state.shared.covered.store(
            if gpu_enabled {
                self.state.fifo.pushed()
            } else {
                u64::MAX
            },
            Ordering::Release,
        );
        self.state.shared.swaps.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::heart::{host_clock, Heart, HeartConfig};
    use crate::dev::heart_mem::HeartRam;
    use crate::traits::Resettable;
    use std::sync::Arc;

    fn board(product: &str) -> Arc<Odyssey> {
        let heart = Arc::new(Heart::new(
            HeartConfig::octane([128 << 20, 0, 0, 0]),
            host_clock(),
        ));
        let ram = Arc::new(HeartRam::try_new([128, 0, 0, 0]).expect("ram"));
        Odyssey::new(
            11,
            product,
            heart,
            ram.dma_view(),
            crate::status::MachineStatus::detached(),
            (2, 32),
            (crate::config::RasterUnit::Cpu, "auto"),
        )
        .expect("VPro board")
    }

    #[test]
    fn byte_parts_reject_truncation_and_trailing_words() {
        let packed = pack_bytes(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(unpack_bytes("test", &packed).unwrap(), [1, 2, 3, 4, 5]);
        assert!(unpack_bytes("test", &packed[..packed.len() - 1]).is_err());
        let mut trailing = packed;
        trailing.push(0);
        assert!(unpack_bytes("test", &trailing).is_err());
    }

    #[test]
    fn state_bound_covers_the_fixed_board_working_sets() {
        assert!(ODYSSEY_STATE_MAX_BYTES > ODSY_WIDTH as usize * ODSY_HEIGHT as usize);
    }

    #[test]
    fn every_vpro_product_round_trips_architectural_state_and_vram() {
        for (index, product, memory_mb) in [
            (1u32, "v6", 32u64),
            (2, "v8", 128),
            (3, "v10", 32),
            (4, "v12", 128),
        ] {
            let vpro = board(product);
            let marker = 0x5100_0000 | index;
            vpro.state.with_stopped_board(|board| {
                assert_eq!(board.vram.size(), memory_mb << 20);
                board.widget.memory_control = marker;
                board.query.stage = index;
                board.query.count = marker;
                board.cmd.state = 1;
                board.cmd.tag = marker ^ 0x1111;
                board.cmd.operand_count = 2;
                board.cmd.operand_index = 1;
                board.cmd.operand[0] = marker ^ 0x2222;
                board.cmd.from_kernel = true;
                board.prim.in_prim = true;
                board.prim.begin_open = true;
                board.prim.restart_armed = true;
                board.prim.open_vertex[0].word = marker ^ 0x3333;
                board.assembly.polygon.vertices =
                    vec![index as f32; super::super::assembly::vtx::FLOATS];
                board.assembly.polygon.count = 1;
                board.tex_mut().path = marker ^ 0x4444;
                board
                    .vram
                    .ptr_shared()
                    .write_words_untracked(0, &[u64::from(marker), !u64::from(marker)]);
            });
            {
                let mut front = vpro.state.front.lock();
                front.sw_token = marker;
                front.swap_arm = 1 << index;
                front.swap_pending = marker ^ 0x5555;
            }
            {
                let mut fifo = vpro.state.display_fifo.lock();
                fifo.index = index;
                fifo.count = 1;
                fifo.ctl[index as usize] = marker ^ 0x6666;
            }

            let (metadata, parts) = vpro.snapshot_parts().unwrap();
            vpro.power_on();
            vpro.restore_snapshot_parts(&metadata, &parts).unwrap();

            vpro.state.with_stopped_board(|board| {
                assert_eq!(board.widget.memory_control, marker, "{product}");
                assert_eq!(
                    (board.query.stage, board.query.count),
                    (index, marker),
                    "{product}"
                );
                assert_eq!(
                    (
                        board.cmd.state,
                        board.cmd.tag,
                        board.cmd.operand_count,
                        board.cmd.operand_index,
                        board.cmd.operand[0],
                        board.cmd.from_kernel,
                    ),
                    (1, marker ^ 0x1111, 2, 1, marker ^ 0x2222, true),
                    "{product}"
                );
                assert!(
                    board.prim.in_prim && board.prim.begin_open && board.prim.restart_armed,
                    "{product}"
                );
                assert_eq!(board.prim.open_vertex[0].word, marker ^ 0x3333, "{product}");
                assert_eq!(board.assembly.polygon.count, 1, "{product}");
                assert_eq!(
                    board.assembly.polygon.vertices,
                    vec![index as f32; super::super::assembly::vtx::FLOATS],
                    "{product}"
                );
                assert_eq!(board.tex_regs().path, marker ^ 0x4444, "{product}");
                assert_eq!(
                    board.vram.ptr_shared().words_at(0, 16),
                    [u64::from(marker), !u64::from(marker)],
                    "{product}"
                );
            });
            let front = vpro.state.front.lock();
            assert_eq!(
                (front.sw_token, front.swap_arm, front.swap_pending),
                (marker, 1 << index, marker ^ 0x5555),
                "{product}"
            );
            drop(front);
            let fifo = vpro.state.display_fifo.lock();
            assert_eq!(
                (fifo.index, fifo.count, fifo.ctl[index as usize]),
                (index, 1, marker ^ 0x6666),
                "{product}"
            );
        }
    }

    #[test]
    fn invalid_or_mismatched_snapshots_do_not_mutate_the_board() {
        let source = board("v6");
        source.state.with_stopped_board(|board| {
            board.widget.memory_control = 0x1111_1111;
            board
                .vram
                .ptr_shared()
                .write_words_untracked(0, &[0x1111_1111_1111_1111]);
        });
        let (metadata, parts) = source.snapshot_parts().unwrap();

        let target = board("v10");
        target.state.with_stopped_board(|board| {
            board.widget.memory_control = 0xaaaa_aaaa;
            board
                .vram
                .ptr_shared()
                .write_words_untracked(0, &[0xaaaa_aaaa_aaaa_aaaa]);
        });
        let error = target
            .restore_snapshot_parts(&metadata, &parts)
            .unwrap_err();
        assert!(error.contains("running board is v10"), "{error}");
        target.state.with_stopped_board(|board| {
            assert_eq!(board.widget.memory_control, 0xaaaa_aaaa);
            assert_eq!(
                board.vram.ptr_shared().words_at(0, 8),
                [0xaaaa_aaaa_aaaa_aaaa]
            );
        });

        let mut bad_metadata: toml::Value = toml::from_str(&metadata).unwrap();
        bad_metadata
            .as_table_mut()
            .unwrap()
            .insert("memory_bytes".into(), toml::Value::Integer(64 << 20));
        let bad_metadata = toml::to_string(&bad_metadata).unwrap();
        source.state.with_stopped_board(|board| {
            board.widget.memory_control = 0xbbbb_bbbb;
            board
                .vram
                .ptr_shared()
                .write_words_untracked(0, &[0xbbbb_bbbb_bbbb_bbbb]);
        });
        let error = source
            .restore_snapshot_parts(&bad_metadata, &parts)
            .unwrap_err();
        assert!(
            error.contains("64") && error.contains("running board"),
            "{error}"
        );
        source.state.with_stopped_board(|board| {
            assert_eq!(board.widget.memory_control, 0xbbbb_bbbb);
            assert_eq!(
                board.vram.ptr_shared().words_at(0, 8),
                [0xbbbb_bbbb_bbbb_bbbb]
            );
        });
    }

    #[test]
    fn named_parts_are_exact_and_power_on_still_resets_restored_state() {
        let vpro = board("v6");
        vpro.state
            .with_stopped_board(|board| board.widget.memory_control = 0x1234_5678);
        let (metadata, mut parts) = vpro.snapshot_parts().unwrap();

        assert!(vpro
            .restore_snapshot_parts(&metadata, &parts[..1])
            .unwrap_err()
            .contains("expected 2"));
        let duplicate = vec![parts[0].clone(), parts[0].clone()];
        assert!(vpro
            .restore_snapshot_parts(&metadata, &duplicate)
            .unwrap_err()
            .contains("duplicate"));
        parts[1].0 = "other".into();
        assert!(vpro
            .restore_snapshot_parts(&metadata, &parts)
            .unwrap_err()
            .contains("vram"));
        parts[1].0 = "vram".into();

        vpro.power_on();
        vpro.restore_snapshot_parts(&metadata, &parts).unwrap();
        vpro.state
            .with_stopped_board(|board| assert_eq!(board.widget.memory_control, 0x1234_5678));
        vpro.power_on();
        vpro.state.with_stopped_board(|board| {
            assert_eq!(board.widget.memory_control, 0);
            assert_eq!(board.vram.ptr_shared().words_at(0, 8), [0]);
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn gpu_restore_reseeds_authoritative_memory() {
        let heart = Arc::new(Heart::new(
            HeartConfig::octane([128 << 20, 0, 0, 0]),
            host_clock(),
        ));
        let ram = Arc::new(HeartRam::try_new([128, 0, 0, 0]).expect("ram"));
        let vpro = match Odyssey::new(
            11,
            "v6",
            heart,
            ram.dma_view(),
            crate::status::MachineStatus::detached(),
            (2, 32),
            (crate::config::RasterUnit::Wgpu, "auto"),
        ) {
            Ok(vpro) => vpro,
            Err(error) => {
                eprintln!("skipped: {error}");
                return;
            }
        };
        let marker = 0x51c2_93a4_b5d6_e7f8;
        vpro.state.with_stopped_board(|board| {
            board.vram.ptr_shared().store(0, 8, marker);
            board.gpu_upload_tick(true);
        });
        let (metadata, parts) = vpro.snapshot_parts().unwrap();
        vpro.power_on();
        vpro.restore_snapshot_parts(&metadata, &parts).unwrap();
        let (mirror, gpu) = vpro.state.with_stopped_board(|board| {
            let mirror = board.vram.ptr_shared().words_at(0, 8)[0];
            let gpu = board.raster.run_now(|ctx| {
                ctx.gpu
                    .as_mut()
                    .expect("GPU unit")
                    .read_words_for(super::super::gpu_unit::Observation::Validation)
                    .map(|words| words[0])
            });
            (mirror, gpu)
        });
        assert_eq!(mirror, marker);
        assert_eq!(gpu.unwrap(), marker);
    }
}
