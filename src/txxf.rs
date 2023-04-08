use anyhow::Result;
use byteorder::ByteOrder;
use nalgebra::{point, Affine2, Matrix3, Point2};
use zerocopy::{FromBytes, U16};

use crate::{brender::Scalar, order::Loader};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureTransform {
    pub min: Point2<f64>,
    pub max: Point2<f64>,
}

impl TextureTransform {
    pub fn transform_point(&self, point: &Point2<f64>) -> Point2<f64> {
        (point.coords.component_mul(&self.size().coords) + self.min.coords).into()
    }

    pub fn size(&self) -> Point2<f64> {
        (self.max - self.min).into()
    }
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self {
            min: Point2::new(0.0, 0.0),
            max: Point2::new(1.0, 1.0),
        }
    }
}

#[derive(FromBytes)]
#[repr(C)]
pub struct TextureTransformOnFile<O>
where
    O: ByteOrder,
{
    byte_order: U16<O>,
    _osk: U16<O>,
    matrix: [[Scalar<O>; 2]; 3],
}

impl<'a> Loader<'a> for TextureTransform {
    type OnFile<O> = TextureTransformOnFile<O>
    where
        O: ByteOrder;

    fn byte_order<O>(on_file: &Self::OnFile<O>) -> u16
    where
        O: ByteOrder,
    {
        on_file.byte_order.get()
    }

    fn into_native<O>(on_file: Self::OnFile<O>, _full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder,
    {
        let matrix = Affine2::from_matrix_unchecked(Matrix3::new(
            on_file.matrix[0][0].into(),
            on_file.matrix[1][0].into(),
            on_file.matrix[2][0].into(),
            on_file.matrix[0][1].into(),
            on_file.matrix[1][1].into(),
            on_file.matrix[2][1].into(),
            0.0,
            0.0,
            1.0,
        ));
        Ok(TextureTransform {
            min: matrix.transform_point(&point![0.0, 0.0]),
            max: matrix.transform_point(&point![1.0, 1.0]),
        })
    }
}
