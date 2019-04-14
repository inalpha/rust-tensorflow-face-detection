use std::error::Error;
use std::env;
use structopt::StructOpt;
use log::debug;

#[derive(StructOpt)]
struct Config {
    #[structopt(
        short = "l",
        long = "listen",
        help = "Listen Address",
        default_value = "127.0.0.1:8000"
    )]
    listen: String,
}




fn main() -> Result<(), Box<dyn Error>> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "rust-tensorflow-face-detection=DEBUG,actix_web=DEBUG")
    }
    pretty_env_logger::init_timed();
    let config = Config::from_args();
    
    debug!("{}", config.listen);

    Ok(())
}
