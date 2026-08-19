//! SGI volume-header image tool: build a PROM-bootable image, or dump one.
//!
//! Build:  mkvh build <out.img> [--size N] [--bootfile NAME]
//!                    [--part slot:type:first:nblks[=file]] <name=path> ...
//! Dump:   mkvh dump <image>
//!
//! A built image is bootable by name from the PROM:
//!   boot -f dksc(0,<scsi-id>,8)<name>
//!
//! Dump mode is the layout check: run it on an original SGI CD and the voldir
//! should list sashARCS and friends with a valid checksum.

use std::path::PathBuf;
use std::process::exit;

use iris::sgi_vh::{build_image, ImageSpec, PartitionSpec, VolumeHeader};

const USAGE: &str = "\
usage:
  mkvh build <out.img> [options] <name=path> [name=path ...]
  mkvh dump  <image>

build options:
  --size N          image size in bytes (K/M/G suffix allowed); default: fit contents
  --bootfile NAME   vh_bootfile; default: the first file
  --part SPEC       partition entry, repeatable:
                    slot:type:first:nblks[=file]  (decimal, or 0x-prefixed)
                    first may be `+` = start right after the volume header
                    e.g. --part 7:5:+:1283016=efs.img
                    NB: SGI ships an EFS filesystem as type 5 (PT_SYSV), not
                    type 7 — that is what the 6.5.22 install CD emits.
";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let r = match args.get(1).map(String::as_str) {
        Some("build") => build(&args[2..]),
        Some("dump") => dump(&args[2..]),
        _ => { eprint!("{}", USAGE); exit(2); }
    };
    if let Err(e) = r {
        eprintln!("mkvh: {}", e);
        exit(1);
    }
}

fn build(args: &[String]) -> Result<(), String> {
    let mut out: Option<PathBuf> = None;
    let mut spec = ImageSpec::default();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--size" => { spec.total_bytes = parse_size(next(args, &mut i, "--size")?)?; }
            "--bootfile" => { spec.bootfile = Some(next(args, &mut i, "--bootfile")?.to_string()); }
            "--part" => { spec.partitions.push(parse_part(next(args, &mut i, "--part")?)?); }
            _ if a.starts_with("--") => return Err(format!("unknown option '{}'\n{}", a, USAGE)),
            _ if out.is_none() => out = Some(PathBuf::from(a)),
            _ => {
                let (name, path) = a.split_once('=')
                    .ok_or_else(|| format!("file argument '{}' is not name=path", a))?;
                spec.files.push((name.to_string(), PathBuf::from(path)));
            }
        }
        i += 1;
    }
    let out = out.ok_or_else(|| format!("no output path given\n{}", USAGE))?;
    if spec.files.is_empty() && spec.partitions.is_empty() {
        return Err(format!("nothing to write\n{}", USAGE));
    }

    let placed = build_image(&out, &spec).map_err(|e| e.to_string())?;
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!("wrote {} ({} bytes)", out.display(), size);
    for f in &placed {
        println!("  {:<8} lbn {:>8}  {:>10} bytes", f.name, f.lbn, f.nbytes);
    }
    Ok(())
}

fn dump(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or_else(|| format!("no image given\n{}", USAGE))?;
    let mut buf = vec![0u8; 512];
    {
        use std::io::Read;
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {}", path, e))?;
        f.read_exact(&mut buf).map_err(|e| format!("{}: {}", path, e))?;
    }
    let vh = VolumeHeader::from_bytes(&buf).map_err(|e| format!("{}: {}", path, e))?;

    println!("{}:", path);
    println!("  magic     0x0BE5A941 (ok)");
    let (rootpt, swappt) = vh.root_swap_parts();
    println!("  rootpt    {}   swappt {}", rootpt, swappt);
    println!("  bootfile  {:?}", vh.bootfile());
    println!("  checksum  {}", if vh.csum_valid() { "valid" } else { "INVALID" });

    let files = vh.files();
    println!("  volume directory ({} of 15 entries):", files.len());
    for f in &files {
        println!("    {:<8} lbn {:>8}  {:>10} bytes", f.name, f.lbn, f.nbytes);
    }

    println!("  partition table:");
    for p in vh.partitions() {
        println!(
            "    slot {:>2}  first {:>10}  nblks {:>10}  type {:>2} ({})",
            p.slot, p.first_block, p.nblks, p.ptype, ptype_name(p.ptype)
        );
    }
    if !vh.csum_valid() {
        return Err("checksum does not validate".to_string());
    }
    Ok(())
}

fn next<'a>(args: &'a [String], i: &mut usize, what: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i).map(String::as_str).ok_or_else(|| format!("{} needs an argument", what))
}

fn parse_num(s: &str) -> Result<u64, String> {
    let t = s.trim();
    let r = match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => t.parse::<u64>(),
    };
    r.map_err(|_| format!("'{}' is not a number", s))
}

fn parse_size(s: &str) -> Result<u64, String> {
    let (num, mult) = match s.chars().last() {
        Some('K') | Some('k') => (&s[..s.len() - 1], 1024),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1024 * 1024),
        Some('G') | Some('g') => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1),
    };
    Ok(parse_num(num)? * mult)
}

fn parse_part(s: &str) -> Result<PartitionSpec, String> {
    let (fields, content) = match s.split_once('=') {
        Some((f, p)) => (f, Some(PathBuf::from(p))),
        None => (s, None),
    };
    let v: Vec<&str> = fields.split(':').collect();
    if v.len() != 4 {
        return Err(format!("--part '{}' must be slot:type:first:nblks[=file]", s));
    }
    let first_block = match v[2] {
        "+" | "auto" => None,
        n => Some(parse_num(n)? as u32),
    };
    Ok(PartitionSpec {
        slot: parse_num(v[0])? as usize,
        ptype: parse_num(v[1])? as u32,
        first_block,
        nblks: parse_num(v[3])? as u32,
        content,
    })
}

fn ptype_name(t: u32) -> &'static str {
    match t {
        0 => "PT_VOLHDR",
        1 => "PT_TRKREPL",
        2 => "PT_SECREPL",
        3 => "PT_RAW",
        4 => "PT_BSD",
        5 => "PT_SYSV", // what SGI media uses for an EFS filesystem
        6 => "PT_VOLUME",
        7 => "PT_EFS",
        8 => "PT_LVOL",
        9 => "PT_RLVOL",
        10 => "PT_XFS",
        11 => "PT_XFSLOG",
        12 => "PT_XLV",
        13 => "PT_XVM",
        _ => "?",
    }
}
