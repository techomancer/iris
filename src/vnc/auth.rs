//! VNC authentication (RFB security type 2): the password and its DES challenge.
//!
//! The server sends 16 random bytes; the viewer returns them DES-encrypted
//! (ECB) with the password as the key. RFB takes the first 8 bytes of the
//! password, zero-padded, and reverses the bits of each key byte. The scheme is
//! weak (8 characters, a known-plaintext challenge, nothing encrypted after
//! it): it keeps casual visitors off a reachable port, nothing more.

use std::io::{Read, Write};
use std::time::Duration;

use des::cipher::generic_array::GenericArray;
use des::cipher::{BlockEncrypt, KeyInit};
use des::Des;

/// Environment variable holding the password itself.
pub const ENV_PASSWORD: &str = "IRIS_VNC_PASSWORD";
/// Environment variable naming a password file, used when `[vnc] password_file`
/// and `--vnc-password-file` are not given.
pub const ENV_PASSWORD_FILE: &str = "IRIS_VNC_PASSWORD_FILE";

/// RFB security type "VNC Authentication".
pub const SEC_VNC_AUTH: u8 = 2;

/// A VNC password, as the DES key RFB derives from it.
#[derive(Clone)]
pub struct Password {
    key: [u8; 8],
}

impl Password {
    /// Only the first 8 bytes count; a longer password is cut (the caller
    /// warns).
    pub fn new(password: &[u8]) -> Password {
        let mut key = [0u8; 8];
        for (k, b) in key.iter_mut().zip(password) {
            *k = b.reverse_bits();
        }
        Password { key }
    }

    fn encrypt(&self, challenge: &[u8; 16]) -> [u8; 16] {
        let des = Des::new(GenericArray::from_slice(&self.key));
        let mut out = *challenge;
        for block in out.chunks_exact_mut(8) {
            des.encrypt_block(GenericArray::from_mut_slice(block));
        }
        out
    }

    fn verify(&self, challenge: &[u8; 16], response: &[u8; 16]) -> bool {
        let want = self.encrypt(challenge);
        // Compare without stopping at the first difference.
        want.iter().zip(response).fold(0u8, |d, (a, b)| d | (a ^ b)) == 0
    }
}

/// Find the password: the file named by `file`, else `IRIS_VNC_PASSWORD_FILE`,
/// else `IRIS_VNC_PASSWORD`. `Ok(None)` when none is set (no authentication).
/// A file's trailing newline is dropped; an empty password is an error rather
/// than an open server by accident.
pub fn load_password(file: Option<&str>) -> Result<Option<Password>, String> {
    let env_file = std::env::var(ENV_PASSWORD_FILE).ok().filter(|v| !v.is_empty());
    let (bytes, from) = if let Some(path) = file.filter(|p| !p.is_empty()).map(str::to_string).or(env_file) {
        let mut b = std::fs::read(&path).map_err(|e| format!("VNC: read password file {path}: {e}"))?;
        if b.ends_with(b"\n") {
            b.pop();
            if b.ends_with(b"\r") {
                b.pop();
            }
        }
        (b, format!("password file {path}"))
    } else if let Some(v) = std::env::var_os(ENV_PASSWORD).filter(|v| !v.is_empty()) {
        (v.to_string_lossy().into_owned().into_bytes(), format!("${ENV_PASSWORD}"))
    } else {
        return Ok(None);
    };
    if bytes.is_empty() {
        return Err(format!("VNC: the {from} is empty"));
    }
    if bytes.len() > 8 {
        eprintln!("iris: VNC: the {from} is longer than 8 bytes; VNC authentication uses only the first 8");
    }
    Ok(Some(Password::new(&bytes)))
}

/// Run the challenge on a viewer that chose type 2, and send the
/// SecurityResult (RFB 3.8 adds a reason on failure).
pub fn exchange<S: Read + Write>(pw: &Password, stream: &mut S, minor: u8) -> Result<(), String> {
    let io = |e: std::io::Error| e.to_string();
    let mut challenge = [0u8; 16];
    getrandom::fill(&mut challenge).map_err(|e| format!("random challenge: {e}"))?;
    stream.write_all(&challenge).map_err(io)?;
    let mut response = [0u8; 16];
    stream.read_exact(&mut response).map_err(io)?;
    if pw.verify(&challenge, &response) {
        return stream.write_all(&0u32.to_be_bytes()).map_err(io);
    }
    // Slow down guessing.
    std::thread::sleep(Duration::from_secs(1));
    let mut fail = 1u32.to_be_bytes().to_vec();
    if minor >= 8 {
        let why = b"authentication failed";
        fail.extend_from_slice(&(why.len() as u32).to_be_bytes());
        fail.extend_from_slice(why);
    }
    let _ = stream.write_all(&fail);
    Err("authentication failed".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};

    #[test]
    fn key_bytes_are_bit_reversed_and_zero_padded() {
        let p = Password::new(b"\x01\x02abc");
        assert_eq!(p.key, [0x80, 0x40, 0x86, 0x46, 0xc6, 0, 0, 0]);
    }

    #[test]
    fn password_is_cut_at_eight_bytes() {
        assert_eq!(Password::new(b"12345678").key, Password::new(b"12345678XYZ").key);
    }

    #[test]
    fn des_known_answer() {
        // The classic worked DES example; the key is given in the usual
        // (non-reversed) order, so undo Password's reversal.
        let key = [0x13, 0x34, 0x57, 0x79, 0x9B, 0xBC, 0xDF, 0xF1];
        let p = Password { key };
        let mut ch = [0u8; 16];
        ch[..8].copy_from_slice(&[0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]);
        assert_eq!(&p.encrypt(&ch)[..8], &[0x85, 0xE8, 0x13, 0x54, 0x0F, 0x0A, 0xB4, 0x05]);
    }

    /// Serve one exchange on a loopback socket; the viewer answers with `answer`.
    fn run(pw: &Password, minor: u8, answer: impl FnOnce([u8; 16]) -> [u8; 16] + Send + 'static) -> (Result<(), String>, Vec<u8>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let viewer = std::thread::spawn(move || {
            let mut c = TcpStream::connect(addr).unwrap();
            let mut ch = [0u8; 16];
            c.read_exact(&mut ch).unwrap();
            c.write_all(&answer(ch)).unwrap();
            let mut rest = Vec::new();
            c.read_to_end(&mut rest).unwrap();
            rest
        });
        let (mut s, _) = l.accept().unwrap();
        let r = exchange(pw, &mut s, minor);
        drop(s);
        (r, viewer.join().unwrap())
    }

    #[test]
    fn right_password_is_accepted() {
        let pw = Password::new(b"secret");
        let same = pw.clone();
        let (r, rest) = run(&pw, 8, move |ch| same.encrypt(&ch));
        assert!(r.is_ok());
        assert_eq!(rest, 0u32.to_be_bytes());
    }

    #[test]
    fn wrong_password_is_refused_with_a_reason() {
        let pw = Password::new(b"secret");
        let other = Password::new(b"Secret");
        let (r, rest) = run(&pw, 8, move |ch| other.encrypt(&ch));
        assert!(r.is_err());
        assert_eq!(&rest[..4], 1u32.to_be_bytes());
        assert_eq!(&rest[8..], b"authentication failed");
    }

    #[test]
    fn rfb_3_7_failure_has_no_reason() {
        let pw = Password::new(b"secret");
        let (r, rest) = run(&pw, 7, |_| [0; 16]);
        assert!(r.is_err());
        assert_eq!(rest, 1u32.to_be_bytes());
    }
}
