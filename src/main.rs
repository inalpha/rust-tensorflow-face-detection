use std::error::Error;

use tensorflow::{Graph, ImportGraphDefOptions, Tensor};
use tensorflow::{Session, SessionOptions, SessionRunArgs};

use std::path::PathBuf;
use structopt::StructOpt;

use image::{GenericImageView, Rgba};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

#[derive(StructOpt)]
struct Config {
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    #[structopt(parse(from_os_str))]
    output: PathBuf,
}

#[derive(Copy, Clone, Debug)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    prob: f32,
}

const LINE_COLOUR: Rgba<u8> = Rgba {
    data: [0, 255, 0, 0],
};

fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::from_args();

    let model = include_bytes!("model.pb");
    let mut graph = Graph::new();
    graph.import_graph_def(&*model, &ImportGraphDefOptions::new())?;

    let img = image::open(&config.input)?;

    let mut flattened = Vec::new();
    for (_x, _y, rgb) in img.pixels() {
        flattened.push(rgb[2] as f32);
        flattened.push(rgb[1] as f32);
        flattened.push(rgb[0] as f32);
    }

    let input =
        Tensor::new(&[img.height() as u64, img.width() as u64, 3]).with_values(&flattened)?;

    let min_size = Tensor::new(&[]).with_values(&[40f32])?;
    let thresholds = Tensor::new(&[3]).with_values(&[0.6f32, 0.7f32, 0.7f32])?;
    let factor = Tensor::new(&[]).with_values(&[0.709f32])?;

    let mut args = SessionRunArgs::new();
    args.add_feed(&graph.operation_by_name_required("min_size")?, 0, &min_size);
    args.add_feed(
        &graph.operation_by_name_required("thresholds")?,
        0,
        &thresholds,
    );
    args.add_feed(&graph.operation_by_name_required("factor")?, 0, &factor);
    args.add_feed(&graph.operation_by_name_required("input")?, 0, &input);

    let bbox = args.request_fetch(&graph.operation_by_name_required("box")?, 0);
    let prob = args.request_fetch(&graph.operation_by_name_required("prob")?, 0);

    let session = Session::new(&SessionOptions::new(), &graph)?;

    session.run(&mut args)?;

    let bbox_res: Tensor<f32> = args.fetch(bbox)?;
    let prob_res: Tensor<f32> = args.fetch(prob)?;

    let mut bboxes = Vec::new();

    let mut i = 0;
    let mut j = 0;

    while i < bbox_res.len() {
        bboxes.push(BBox {
            y1: bbox_res[i],
            x1: bbox_res[i + 1],
            y2: bbox_res[i + 2],
            x2: bbox_res[i + 3],
            prob: prob_res[j],
        });
        i += 4;
        j += 1;
    }

    println!("BBox Length: {}", bboxes.len());
    let mut output = img.clone();

    for bbox in bboxes {
        let rect = Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32);
        draw_hollow_rect_mut(&mut output, rect, LINE_COLOUR)
    }   

    output.save(&config.output)?;
    Ok(())
}
