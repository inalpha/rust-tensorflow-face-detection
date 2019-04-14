use serde_derive::Serialize;
use std::error::Error;


use tensorflow::{Graph, ImportGraphDefOptions, Tensor, Status};
use tensorflow::{Session, SessionOptions, SessionRunArgs};

use image::{DynamicImage, GenericImageView, Rgba};

use log::debug;

pub struct MTCNN {
    graph: Graph,
    session: Session,
    min_size: Tensor<f32>,
    thresholds: Tensor<f32>,
    factor: Tensor<f32>,
}

#[derive(Copy, Clone, Debug, Serialize)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
}

const LINE_COLOUR: Rgba<u8> = Rgba {
    data: [0, 255, 0, 0],
};

impl MTCNN {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let model = include_bytes!("model.pb");
        let mut graph = Graph::new();
        graph.import_graph_def(&*model, &ImportGraphDefOptions::new())?;

        let session = Session::new(&SessionOptions::new(), &graph)?;

        let min_size = Tensor::new(&[]).with_values(&[40.0])?;
        let thresholds = Tensor::new(&[3]).with_values(&[0.6, 0.7, 0.7])?;
        let factor = Tensor::new(&[]).with_values(&[0.709])?;

        Ok(Self {
            graph,
            session,
            min_size,
            thresholds,
            factor,
        })
    }
    pub fn run(&self, img: &DynamicImage) -> Result<Vec<BBox>, Status> {
        let input = {
            let mut flattened = Vec::new();

            for (_x, _y, rgb) in img.pixels() {
                flattened.push(rgb[2] as f32);
                flattened.push(rgb[1] as f32);
                flattened.push(rgb[0] as f32);
            }

            Tensor::new(&[img.height() as u64, img.width() as u64, 3]).with_values(&flattened)?
        };

        let mut args = SessionRunArgs::new();
        args.add_feed(
            &self.graph.operation_by_name_required("min_size")?,
            0,
            &self.min_size,
        );
        args.add_feed(
            &self.graph.operation_by_name_required("thresholds")?,
            0,
            &self.thresholds,
        );
        args.add_feed(
            &self.graph.operation_by_name_required("factor")?,
            0,
            &self.factor,
        );
        args.add_feed(&self.graph.operation_by_name_required("input")?, 0, &input);

        let bbox = args.request_fetch(&self.graph.operation_by_name_required("box")?, 0);
        let prob = args.request_fetch(&self.graph.operation_by_name_required("prob")?, 0);

        &self.session.run(&mut args)?;

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

        debug!("BBox Length: {}", bboxes.len());

        Ok(bboxes)
    }
}
pub fn overlay(img: &DynamicImage, bboxs: &Vec<BBox>) -> DynamicImage {
    use imageproc::drawing::draw_hollow_rect_mut;
    use imageproc::rect::Rect;

    let mut output = img.clone();
    for bbox in bboxs {
        let rect = Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32);;
        draw_hollow_rect_mut(&mut output, rect, LINE_COLOUR);
    }
    output
}
