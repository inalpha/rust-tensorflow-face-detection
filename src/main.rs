use std::env;
use std::error::Error;
use std::sync::Arc;

use log::info;
use structopt::StructOpt;

mod mtcnn;
use mtcnn::{BBox, MTCNN};

use actix_web::{middleware, web, App, Error as ActixError, HttpResponse, HttpServer};
use futures::{Future, Stream};

use image::DynamicImage;

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

type WebMTCNN = web::Data<Arc<MTCNN>>;

fn main() -> Result<(), Box<dyn Error>> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "mtcnn=DEBUG,actix_web=DEBUG")
    }
    pretty_env_logger::init_timed();
    let config = Config::from_args();
    info!("Listening on: {}", config.listen);

    let mtcnn = Arc::new(MTCNN::new()?);

    Ok(HttpServer::new(move || {
        App::new()
            .data(mtcnn.clone())
            .wrap(middleware::Logger::default())
            .service(web::resource("api/v1/bboxes").to_async(handle_bboxes))
    })
    .bind(&config.listen)?
    .run()?)
}

fn get_image(stream: web::Payload) -> impl Future<Item = DynamicImage, Error = ActixError> {
    stream
        .concat2()
        .from_err()
        .and_then(move |bytes| web::block(move || image::load_from_memory(&bytes)).from_err())
}

fn get_bboxes(
    img: DynamicImage,
    mtcnn: WebMTCNN,
) -> impl Future<Item = Vec<BBox>, Error = ActixError> {
    web::block(move || mtcnn.run(&img).map_err(|e| e.to_string())).from_err()
}

fn handle_bboxes(
    stream: web::Payload,
    mtcnn: WebMTCNN,
) -> impl Future<Item = HttpResponse, Error = ActixError> {
    get_image(stream)
        .and_then(move |img| get_bboxes(img, mtcnn))
        .map(|bboxes| HttpResponse::Ok().json(bboxes))
}
