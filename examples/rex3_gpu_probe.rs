fn main() {
    let backend = std::env::var("REX3_GPU_BACKEND").unwrap_or_else(|_| "auto".into());
    match iris::dev::ng1::rex3_gpu::probe(&backend) {
        Ok(result) => {
            println!("{}", result.diagnostics);
            println!("{:?}", result.report);
        }
        Err(error) => {
            eprintln!("REX3 GPU probe failed: {error}");
            std::process::exit(1);
        }
    }
}
