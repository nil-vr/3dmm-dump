use anyhow::{bail, Result};
use byteorder::ByteOrder;
use nalgebra::{Affine3, Matrix4};
use zerocopy::FromBytes;

use crate::{
    brender::Scalar,
    glf::{List, ListOnFile},
    order::Loader,
};

#[derive(FromBytes)]
#[repr(C)]
struct Mat34OnFile<O>
where
    O: ByteOrder,
{
    m: [[Scalar<O>; 3]; 4],
}

pub struct AnimationTransforms {
    pub transforms: Vec<Affine3<f64>>,
}

impl<'a> Loader<'a> for AnimationTransforms {
    type OnFile<O> = ListOnFile<O>
    where
        O: ByteOrder;

    fn byte_order<O>(on_file: &Self::OnFile<O>) -> u16
    where
        O: ByteOrder,
    {
        List::byte_order(on_file)
    }

    fn into_native<O>(on_file: Self::OnFile<O>, full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder,
    {
        let list = List::from_file(&on_file, full_input)?;

        let mut transforms = Vec::with_capacity(list.len());
        for v in list.iter() {
            let Some(v) = Mat34OnFile::<O>::read_from(v) else {
                bail!("Invalid list item size");
            };
            transforms.push(Affine3::from_matrix_unchecked(Matrix4::new(
                v.m[0][0].into(),
                v.m[1][0].into(),
                v.m[2][0].into(),
                v.m[3][0].into(),
                v.m[0][1].into(),
                v.m[1][1].into(),
                v.m[2][1].into(),
                v.m[3][1].into(),
                v.m[0][2].into(),
                v.m[1][2].into(),
                v.m[2][2].into(),
                v.m[3][2].into(),
                0.0,
                0.0,
                0.0,
                1.0,
            )));
        }

        Ok(AnimationTransforms { transforms })
    }
}
