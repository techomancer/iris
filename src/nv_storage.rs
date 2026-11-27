//! Replace battery-backed state without truncating the previous image first.
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) fn save(filename: &str, bytes: &[u8]) -> io::Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = Path::new(filename);
    let name = path.file_name().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing filename"))?;
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut temp_name = name.to_os_string();
    temp_name.push(format!(".{}.{}.tmp", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let temp = parent.join(temp_name);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp)?;
    let result = (|| {
        if let Ok(metadata) = fs::metadata(path) {
            file.set_permissions(metadata.permissions())?;
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)
    })();
    if result.is_err() { let _ = fs::remove_file(temp); }
    result
}

/// `headless` stores `console=d` through `set` and says where; without it, a
/// store that has `console=d` gets a line saying why the PROM uses the serial port.
pub(crate) fn apply_headless_console(headless: bool, machine: &str, path: &str, serial: bool, set: impl FnOnce()) {
    if headless && !serial {
        set();
        eprintln!("iris: {machine}: headless stored console=d in {path} (setenv console g to change)");
    } else if !headless && serial {
        eprintln!("iris: {machine} NVRAM {path} has console=d: the PROM uses the serial console (setenv console g to change)");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn failed_replacement_preserves_target_and_removes_temporary_file() {
        let dir = std::env::temp_dir().join(format!("iris-nv-save-failure-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let target = dir.join("nvram.bin");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("original"), b"previous state").unwrap();
        assert!(super::save(target.to_str().unwrap(), b"new state").is_err());
        assert_eq!(std::fs::read(target.join("original")).unwrap(), b"previous state");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
