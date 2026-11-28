//! The host-camera test: opens the camera the VINO / IndyCam emulation would
//! use and shows its picture, so the camera (and its permission prompt) can be
//! checked without booting IRIX.

use iris::config::{MachineConfig, VinoStandard};
use iris::video_source::VideoStandard;

use crate::model::camera_test::CameraTest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraAction {
    /// Open the camera the selected machine is set to use.
    Open,
    /// Close the preview and release the camera.
    Close,
}

#[derive(Default)]
pub struct CameraTestController {
    test: Option<CameraTest>,
}

impl CameraTestController {
    pub fn is_open(&self) -> bool {
        self.test.is_some()
    }

    /// The status line under the preview, and whether it is an error.
    pub fn status(&self) -> Option<(String, bool)> {
        let t = self.test.as_ref()?;
        Some(match t.error() {
            Some(e) => (e, true),
            None => (t.status(), false),
        })
    }

    /// A frame newer than `last_seq`: width, height and RGBA bytes.
    pub fn take_new_frame(&self, last_seq: &mut u64) -> Option<(u32, u32, Vec<u8>)> {
        self.test.as_ref()?.take_new_frame(last_seq)
    }

    pub(crate) fn apply(&mut self, action: CameraAction, cfg: &MachineConfig) {
        // Drop any previous instance first, so the camera is released before
        // it is opened again (its Drop joins the capture thread).
        self.test = None;
        if action == CameraAction::Open {
            let standard = match cfg.vino.standard {
                VinoStandard::Ntsc => VideoStandard::Ntsc,
                VinoStandard::Pal => VideoStandard::Pal,
            };
            self.test = Some(CameraTest::start(standard, cfg.vino.camera_index));
        }
    }
}
