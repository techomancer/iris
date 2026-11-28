// Embed every `locales/<language>/*.ftl` catalog (see `src/model/i18n.rs`).
// Adding a language is adding a folder under iris-gui/locales; nothing in the
// source needs to list it. Every shell shares these catalogs.
fn main() {
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../locales");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut languages: Vec<_> = std::fs::read_dir(&root)
        .expect("iris-gui/locales")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    languages.sort();
    let mut out = String::from("pub static LOCALES: &[(&str, &[(&str, &str)])] = &[\n");
    for dir in languages {
        println!("cargo:rerun-if-changed={}", dir.display());
        let lang = dir.file_name().unwrap().to_string_lossy().into_owned();
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "ftl"))
            .collect();
        files.sort();
        out.push_str(&format!("    ({lang:?}, &[\n"));
        for f in files {
            println!("cargo:rerun-if-changed={}", f.display());
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            let path = dunce_free_canonical(&f);
            out.push_str(&format!("        ({name:?}, include_str!({path:?})),\n"));
        }
        out.push_str("    ]),\n");
    }
    out.push_str("];\n");
    let dest = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("locales.rs");
    std::fs::write(dest, out).unwrap();
}

/// An absolute path without `..`, for `include_str!`.
fn dunce_free_canonical(p: &std::path::Path) -> String {
    let c = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let s = c.display().to_string();
    // Windows canonical paths carry a `\\?\` prefix include_str! accepts, but strip
    // it so the generated file reads like the others.
    s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
}
