use std::error::Error; 

use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Config {
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    #[structopt(parse(from_os_str))]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let model = include_bytes!("model.pb");
    let config = Config::from_args();
    let mut graph = Graph::new();
    graph.import_graph_def(&*model, &ImportGraphDefOptions::new())?;

    Ok(())
}