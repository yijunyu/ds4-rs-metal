use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: inspect_gguf <gguf>")?;
    let filter = env::args().nth(2).unwrap_or_default();
    let g = ds4_engine::gguf::GgufFile::open(Path::new(&path))?;
    println!("n_tensors={}", g.tensors.len());
    for t in &g.tensors {
        if filter.is_empty() || t.name.contains(&filter) {
            println!("{} type={:?} dims={:?}", t.name, t.ttype, t.dims);
        }
    }
    Ok(())
}
