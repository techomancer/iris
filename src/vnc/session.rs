//! One viewer: the RFB handshake, its messages in, its updates out.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use winit::keyboard::KeyCode;

use super::encode::{self, PixelFormat, Rect};
use super::{Frame, Shared, TILE};
use crate::hostinput::{BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT};

const ENC_RAW: i32 = 0;
const ENC_ZRLE: i32 = 16;
const ENC_DESKTOP_SIZE: i32 = -223;
const ENC_QEMU_EXT_KEY: i32 = -258;

#[derive(Clone, Copy)]
struct Request {
    incremental: bool,
    rect: Rect,
}

struct Client {
    pf: PixelFormat,
    zrle: bool,
    desktop_size: bool,
    ext_key: bool,
    ext_key_acked: bool,
    request: Option<Request>,
    seen_gen: u64,
    size_gen: u64,
    dead: Option<String>,
    held_keys: HashSet<KeyCode>,
    /// Viewer buttons (RFB bits) last seen, for wheel edges.
    vnc_buttons: u8,
}

pub(crate) fn serve(s: &Arc<Shared>, mut stream: TcpStream) -> Result<(), String> {
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
    let io = |e: std::io::Error| e.to_string();

    // Version, security (None), ClientInit.
    stream.write_all(b"RFB 003.008\n").map_err(io)?;
    let mut ver = [0u8; 12];
    stream.read_exact(&mut ver).map_err(io)?;
    let minor = match &ver {
        b"RFB 003.003\n" => 3,
        b"RFB 003.007\n" => 7,
        v if v.starts_with(b"RFB 003.") => 8,
        _ => return Err(format!("not an RFB viewer ({:?})", String::from_utf8_lossy(&ver))),
    };
    if minor == 3 {
        stream.write_all(&1u32.to_be_bytes()).map_err(io)?;
    } else {
        stream.write_all(&[1, 1]).map_err(io)?;
        let mut choice = [0u8];
        stream.read_exact(&mut choice).map_err(io)?;
        if choice[0] != 1 {
            return Err(format!("viewer chose security type {}", choice[0]));
        }
        if minor >= 8 {
            stream.write_all(&0u32.to_be_bytes()).map_err(io)?;
        }
    }
    let mut shared_flag = [0u8];
    stream.read_exact(&mut shared_flag).map_err(io)?;

    // The size to announce: wait a little for the first frame.
    let frame = {
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut f = s.frame.lock();
        while f.w == 0 && Instant::now() < deadline {
            s.changed.wait_for(&mut f, Duration::from_millis(100));
        }
        f.clone()
    };
    let (w, h) = if frame.w > 0 { (frame.w, frame.h) } else { (1024, 768) };
    let mut init = Vec::new();
    init.extend_from_slice(&(w as u16).to_be_bytes());
    init.extend_from_slice(&(h as u16).to_be_bytes());
    init.extend_from_slice(&PixelFormat::NATIVE.to_bytes());
    init.extend_from_slice(&(s.name.len() as u32).to_be_bytes());
    init.extend_from_slice(s.name.as_bytes());
    stream.write_all(&init).map_err(io)?;
    stream.set_read_timeout(None).ok();

    let client = Arc::new(Mutex::new(Client {
        pf: PixelFormat::NATIVE,
        zrle: false,
        desktop_size: false,
        ext_key: false,
        ext_key_acked: false,
        request: None,
        seen_gen: 0,
        // A placeholder size is a size change still to send.
        size_gen: if frame.w > 0 { frame.size_gen } else { u64::MAX },
        dead: None,
        held_keys: HashSet::new(),
        vnc_buttons: 0,
    }));

    let reader = {
        let (s, client) = (s.clone(), client.clone());
        let mut rd = stream.try_clone().map_err(io)?;
        std::thread::Builder::new().name("VNC-Read".into()).spawn(move || {
            let why = read_loop(&s, &client, &mut rd).err().unwrap_or_else(|| "closed".into());
            client.lock().dead = Some(why);
            s.changed.notify_all();
        }).map_err(io)?
    };

    let result = send_loop(s, &client, &mut stream);
    let _ = stream.shutdown(Shutdown::Both);
    let _ = reader.join();

    // Let go of whatever this viewer was holding.
    let c = client.lock();
    for &k in &c.held_keys {
        s.input.key(k, false);
    }
    if c.vnc_buttons & 7 != 0 {
        s.input.set_buttons(0);
    }
    match result {
        Err(e) => Err(e),
        Ok(()) => match &c.dead {
            Some(why) if why != "closed" => Err(why.clone()),
            _ => Ok(()),
        },
    }
}

fn read_loop(s: &Shared, client: &Mutex<Client>, rd: &mut TcpStream) -> Result<(), String> {
    let mut b = [0u8; 20];
    loop {
        let mut ty = [0u8];
        if rd.read(&mut ty).map_err(|e| e.to_string())? == 0 {
            return Ok(());
        }
        match ty[0] {
            0 => {
                let m = take(rd, &mut b, 19)?;
                let pf = PixelFormat::from_bytes(m[3..19].try_into().unwrap())?;
                client.lock().pf = pf;
            }
            2 => {
                let m = take(rd, &mut b, 3)?;
                let n = u16::from_be_bytes([m[1], m[2]]) as usize;
                let mut list = vec![0u8; n * 4];
                rd.read_exact(&mut list).map_err(|e| e.to_string())?;
                let encs: Vec<i32> = list.chunks(4).map(|c| i32::from_be_bytes(c.try_into().unwrap())).collect();
                let mut c = client.lock();
                c.zrle = encs.contains(&ENC_ZRLE);
                c.desktop_size = encs.contains(&ENC_DESKTOP_SIZE);
                let ext = encs.contains(&ENC_QEMU_EXT_KEY);
                if ext && !c.ext_key {
                    c.ext_key_acked = false;
                }
                c.ext_key = ext;
            }
            3 => {
                let m = take(rd, &mut b, 9)?;
                let v = |i: usize| u16::from_be_bytes([m[i], m[i + 1]]) as usize;
                client.lock().request = Some(Request {
                    incremental: m[0] != 0,
                    rect: Rect { x: v(1), y: v(3), w: v(5), h: v(7) },
                });
                s.changed.notify_all();
            }
            4 => {
                let m = take(rd, &mut b, 7)?;
                let sym = u32::from_be_bytes(m[3..7].try_into().unwrap());
                key(s, client, m[0] != 0, crate::keysym::from_keysym(sym));
            }
            5 => {
                let m = take(rd, &mut b, 5)?;
                let (mask, x, y) = (m[0], u16::from_be_bytes([m[1], m[2]]) as i32, u16::from_be_bytes([m[3], m[4]]) as i32);
                let rising = {
                    let mut c = client.lock();
                    let r = mask & !c.vnc_buttons;
                    c.vnc_buttons = mask;
                    r
                };
                // Wheel up and down come as presses of buttons 4 and 5.
                if rising & 8 != 0 {
                    s.input.wheel(1);
                }
                if rising & 16 != 0 {
                    s.input.wheel(-1);
                }
                let mut ps2 = 0;
                if mask & 1 != 0 { ps2 |= BUTTON_LEFT; }
                if mask & 2 != 0 { ps2 |= BUTTON_MIDDLE; }
                if mask & 4 != 0 { ps2 |= BUTTON_RIGHT; }
                s.pointer.push(x, y, ps2);
            }
            6 => {
                let m = take(rd, &mut b, 7)?;
                let n = u32::from_be_bytes(m[3..7].try_into().unwrap()) as u64;
                // Clipboard: not passed on.
                std::io::copy(&mut Read::by_ref(rd).take(n), &mut std::io::sink()).map_err(|e| e.to_string())?;
            }
            255 => {
                let m = take(rd, &mut b, 1)?;
                if m[0] != 0 {
                    return Err(format!("unknown QEMU message {}", m[0]));
                }
                let m = take(rd, &mut b, 10)?;
                let down = u16::from_be_bytes([m[0], m[1]]) != 0;
                let sym = u32::from_be_bytes(m[2..6].try_into().unwrap());
                let xt = u32::from_be_bytes(m[6..10].try_into().unwrap());
                let code = crate::keysym::from_xt(xt).or_else(|| crate::keysym::from_keysym(sym));
                key(s, client, down, code);
            }
            t => return Err(format!("unknown message type {t}")),
        }
    }
}

fn take<'a>(rd: &mut TcpStream, b: &'a mut [u8; 20], n: usize) -> Result<&'a [u8], String> {
    rd.read_exact(&mut b[..n]).map_err(|e| e.to_string())?;
    Ok(&b[..n])
}

fn key(s: &Shared, client: &Mutex<Client>, down: bool, code: Option<KeyCode>) {
    let Some(code) = code else { return };
    let mut c = client.lock();
    if down {
        c.held_keys.insert(code);
    } else {
        c.held_keys.remove(&code);
    }
    drop(c);
    s.input.key(code, down);
}

/// What to send next, or None to keep waiting.
fn next_update(f: &Frame, c: &mut Client) -> Result<Option<(Vec<(Rect, i32)>, Vec<Rect>)>, String> {
    let Some(req) = c.request else { return Ok(None) };
    if f.w == 0 {
        return Ok(None);
    }
    let mut pseudo = Vec::new();
    if c.ext_key && !c.ext_key_acked {
        pseudo.push((Rect { x: 0, y: 0, w: 0, h: 0 }, ENC_QEMU_EXT_KEY));
        c.ext_key_acked = true;
    }
    let full = Rect { x: 0, y: 0, w: f.w, h: f.h };
    let rects = if c.size_gen != f.size_gen {
        if !c.desktop_size {
            return Err(format!("the screen is now {}x{} and the viewer cannot resize: reconnect", f.w, f.h));
        }
        pseudo.push((full, ENC_DESKTOP_SIZE));
        vec![full]
    } else if !req.incremental {
        clip(req.rect, f).into_iter().collect()
    } else if f.gen > c.seen_gen {
        dirty_rects(f, c.seen_gen, req.rect)
    } else {
        Vec::new()
    };
    c.seen_gen = f.gen;
    c.size_gen = f.size_gen;
    if rects.is_empty() && pseudo.is_empty() {
        // Nothing in the requested area changed; the request stands.
        return Ok(None);
    }
    c.request = None;
    Ok(Some((pseudo, rects)))
}

fn clip(r: Rect, f: &Frame) -> Option<Rect> {
    let (x1, y1) = ((r.x + r.w).min(f.w), (r.y + r.h).min(f.h));
    (r.x < x1 && r.y < y1).then(|| Rect { x: r.x, y: r.y, w: x1 - r.x, h: y1 - r.y })
}

/// Tiles changed since `seen` inside `area`, merged into runs across and
/// stacked runs down.
fn dirty_rects(f: &Frame, seen: u64, area: Rect) -> Vec<Rect> {
    let tiles_y = f.h.div_ceil(TILE);
    let mut out: Vec<Rect> = Vec::new();
    // Rects that ended on the previous tile row, by index into `out`.
    let mut open: Vec<usize> = Vec::new();
    for ty in 0..tiles_y {
        let mut next_open = Vec::new();
        let mut tx = 0;
        while tx < f.tiles_x {
            if f.tile_gen[ty * f.tiles_x + tx] <= seen {
                tx += 1;
                continue;
            }
            let start = tx;
            while tx < f.tiles_x && f.tile_gen[ty * f.tiles_x + tx] > seen {
                tx += 1;
            }
            let r = Rect {
                x: start * TILE,
                y: ty * TILE,
                w: (tx * TILE).min(f.w) - start * TILE,
                h: ((ty + 1) * TILE).min(f.h) - ty * TILE,
            };
            match open.iter().position(|&i| out[i].x == r.x && out[i].w == r.w) {
                Some(p) => {
                    let i = open.remove(p);
                    out[i].h += r.h;
                    next_open.push(i);
                }
                None => {
                    out.push(r);
                    next_open.push(out.len() - 1);
                }
            }
        }
        open = next_open;
    }
    out.into_iter().filter_map(|r| {
        let x0 = r.x.max(area.x);
        let y0 = r.y.max(area.y);
        let x1 = (r.x + r.w).min(area.x + area.w);
        let y1 = (r.y + r.h).min(area.y + area.h);
        (x0 < x1 && y0 < y1).then(|| Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
    }).collect()
}

fn send_loop(s: &Shared, client: &Mutex<Client>, stream: &mut TcpStream) -> Result<(), String> {
    let mut zrle = encode::Zrle::new();
    let mut buf = Vec::new();
    loop {
        let (frame, pseudo, rects, pf, use_zrle) = {
            let mut f = s.frame.lock();
            loop {
                if s.stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let mut c = client.lock();
                if c.dead.is_some() {
                    return Ok(());
                }
                if let Some((pseudo, rects)) = next_update(&f, &mut c)? {
                    break (f.clone(), pseudo, rects, c.pf, c.zrle);
                }
                drop(c);
                s.changed.wait_for(&mut f, Duration::from_millis(100));
            }
        };
        buf.clear();
        buf.extend_from_slice(&[0, 0]);
        buf.extend_from_slice(&((pseudo.len() + rects.len()) as u16).to_be_bytes());
        let header = |buf: &mut Vec<u8>, r: Rect, enc: i32| {
            for v in [r.x, r.y, r.w, r.h] {
                buf.extend_from_slice(&(v as u16).to_be_bytes());
            }
            buf.extend_from_slice(&enc.to_be_bytes());
        };
        for &(r, enc) in &pseudo {
            header(&mut buf, r, enc);
        }
        for &r in &rects {
            if use_zrle {
                header(&mut buf, r, ENC_ZRLE);
                zrle.encode(&pf, &frame.px, frame.w, r, &mut buf)?;
            } else {
                header(&mut buf, r, ENC_RAW);
                encode::raw(&pf, &frame.px, frame.w, r, &mut buf);
            }
        }
        stream.write_all(&buf).map_err(|e| e.to_string())?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, tile_gen: Vec<u64>) -> Frame {
        Frame {
            w, h, px: Arc::new(vec![0; w * h]), tiles_x: w.div_ceil(TILE),
            tile_gen: Arc::new(tile_gen), gen: 5, size_gen: 0,
        }
    }

    #[test]
    fn dirty_tiles_merge() {
        // 3x2 tiles, 150x100: a 2-tile run on both rows (stacks), and one tile.
        let f = frame(150, 100, vec![5, 5, 0, 5, 5, 4]);
        let all = Rect { x: 0, y: 0, w: 150, h: 100 };
        let r = dirty_rects(&f, 4, all);
        assert_eq!(r, vec![Rect { x: 0, y: 0, w: 128, h: 100 }]);
        let r = dirty_rects(&f, 3, all);
        assert_eq!(r, vec![Rect { x: 0, y: 0, w: 128, h: 64 }, Rect { x: 0, y: 64, w: 150, h: 36 }]);
        // Clipped to the requested area.
        let r = dirty_rects(&f, 4, Rect { x: 10, y: 10, w: 20, h: 20 });
        assert_eq!(r, vec![Rect { x: 10, y: 10, w: 20, h: 20 }]);
        assert!(dirty_rects(&f, 5, all).is_empty());
    }
}
