//! WGSL assembled from source files on disk.
//!
//! The kernels are kept as `.wgsl` files next to the code that uses them
//! and compiled into the binary with `include_str!`. WGSL has no include,
//! so a `Source` is a list of named fragments; a line `#include "name"`
//! splices the fragment of that name in place, once. Constants both sides
//! must agree on (tile size, mask words) come in through `define`, which
//! prepends `const NAME: u32 = value;`. Every fragment is preceded by a
//! comment naming it, so the line numbers in naga's errors can be read
//! back to a file.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// A named piece of WGSL, usually `Fragment::new("x.wgsl", include_str!("x.wgsl"))`.
#[derive(Clone, Copy, Debug)]
pub struct Fragment {
    pub name: &'static str,
    pub text: &'static str,
}

impl Fragment {
    pub const fn new(name: &'static str, text: &'static str) -> Fragment {
        Fragment { name, text }
    }
}

/// A shader under assembly.
#[derive(Clone, Debug, Default)]
pub struct Source {
    label: String,
    defines: Vec<(String, String)>,
    /// Fragments by name, for `#include`.
    library: BTreeMap<&'static str, &'static str>,
    /// The fragments that form the module body, in order.
    body: Vec<Fragment>,
}

impl Source {
    pub fn new(label: &str) -> Source {
        Source { label: label.to_owned(), ..Default::default() }
    }

    /// A fragment `#include` can name. Including is not emitting.
    pub fn library(mut self, f: Fragment) -> Source {
        self.library.insert(f.name, f.text);
        self
    }

    /// A fragment of the module body.
    pub fn body(mut self, f: Fragment) -> Source {
        self.library.insert(f.name, f.text);
        self.body.push(f);
        self
    }

    /// `const NAME: u32 = value;` at the top.
    pub fn define(mut self, name: &str, value: u32) -> Source {
        self.defines.push((name.to_owned(), format!("{value}u")));
        self
    }

    /// `const NAME: i32 = value;` at the top.
    pub fn define_i32(mut self, name: &str, value: i32) -> Source {
        self.defines.push((name.to_owned(), format!("{value}i")));
        self
    }

    /// The assembled WGSL.
    pub fn text(&self) -> Result<String, String> {
        let mut out = String::new();
        for (name, value) in &self.defines {
            let ty = if value.ends_with('i') { "i32" } else { "u32" };
            let _ = writeln!(out, "const {name}: {ty} = {value};");
        }
        let mut included = Vec::new();
        for f in &self.body {
            self.splice(f.name, f.text, &mut out, &mut included, 0)?;
        }
        Ok(out)
    }

    fn splice(
        &self,
        name: &str,
        text: &str,
        out: &mut String,
        included: &mut Vec<String>,
        depth: usize,
    ) -> Result<(), String> {
        if depth > 16 {
            return Err(format!("{}: include nesting too deep at {name}", self.label));
        }
        if included.iter().any(|n| n == name) {
            return Ok(());
        }
        included.push(name.to_owned());
        let _ = writeln!(out, "// ---- {name}");
        for line in text.lines() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("#include") {
                let inc = rest.trim().trim_matches('"');
                let Some(body) = self.library.get(inc) else {
                    return Err(format!("{}: {name} includes unknown fragment {inc:?}", self.label));
                };
                self.splice(inc, body, out, included, depth + 1)?;
                let _ = writeln!(out, "// ---- {name} (continued)");
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        Ok(())
    }

    /// Compile. A validation error comes back as text with the composed
    /// source's line numbers, not as a panic.
    pub fn compile(&self, device: &wgpu::Device) -> Result<wgpu::ShaderModule, String> {
        let mut text = self.text()?;
        let name = self.label.replace(['/', ' '], "_");
        // IRIS_WGPU_DUMP=<dir>: the composed WGSL of every kernel, for
        // offline tools.
        if let Ok(dir) = std::env::var("IRIS_WGPU_DUMP") {
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(format!("{dir}/{name}.wgsl"), &text);
        }
        // IRIS_WGPU_LOAD=<dir>: compile <dir>/<kernel>.wgsl instead, where
        // there is one (a dumped kernel, edited), to try a change without a
        // rebuild.
        if let Ok(dir) = std::env::var("IRIS_WGPU_LOAD") {
            if let Ok(t) = std::fs::read_to_string(format!("{dir}/{name}.wgsl")) {
                log::warn!("wgpu: {} loaded from {dir}", self.label);
                text = t;
            }
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&self.label),
            source: wgpu::ShaderSource::Wgsl(text.as_str().into()),
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("{}: {e}\n{}", self.label, numbered(&text)));
        }
        Ok(module)
    }

    /// A compute pipeline for `entry` in this source, with the given bind
    /// group layouts.
    pub fn compute_pipeline(
        &self,
        device: &wgpu::Device,
        entry: &str,
        layouts: &[&wgpu::BindGroupLayout],
    ) -> Result<wgpu::ComputePipeline, String> {
        let module = self.compile(device)?;
        let groups: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&self.label),
            bind_group_layouts: &groups,
            immediate_size: 0,
        });
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}::{entry}", self.label)),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("{}::{entry}: {e}", self.label));
        }
        Ok(pipeline)
    }
}

fn numbered(text: &str) -> String {
    let mut s = String::new();
    for (i, l) in text.lines().enumerate() {
        let _ = writeln!(s, "{:5} {l}", i + 1);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_splice_once_and_defines_lead() {
        let s = Source::new("t")
            .define("TILE_W", 8)
            .library(Fragment::new("a.wgsl", "fn a() -> u32 { return 1u; }"))
            .body(Fragment::new("m.wgsl", "#include \"a.wgsl\"\n#include \"a.wgsl\"\nfn m() -> u32 { return a() + TILE_W; }"));
        let t = s.text().unwrap();
        assert!(t.starts_with("const TILE_W: u32 = 8u;\n"));
        assert_eq!(t.matches("fn a()").count(), 1);
        assert!(t.contains("// ---- a.wgsl"));
    }

    #[test]
    fn unknown_include_is_an_error() {
        let s = Source::new("t").body(Fragment::new("m.wgsl", "#include \"nope.wgsl\""));
        assert!(s.text().unwrap_err().contains("nope.wgsl"));
    }
}
