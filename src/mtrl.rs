use anyhow::Result;
use byteorder::ByteOrder;
use zerocopy::{FromBytes, U16, U32};

use crate::{
    brender::{Scalar, UFraction},
    order::Loader,
};

#[derive(FromBytes)]
#[repr(C)]
pub struct MaterialOnFile<O>
where
    O: ByteOrder,
{
    byte_order: U16<O>,
    _osk: U16<O>,
    _color: U32<O>,
    ambient: UFraction<O>,
    diffuse: UFraction<O>,
    specular: UFraction<O>,
    index_base: u8,
    _index_len: u8,
    specular_exponent: Scalar<O>,
}

pub struct Material {
    pub color: u8,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub specular_exponent: f64,
}

impl<'a> Loader<'a> for Material {
    type OnFile<O> = MaterialOnFile<O>
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
        Ok(Material {
            ambient: on_file.ambient.into(),
            color: on_file.index_base,
            diffuse: on_file.diffuse.into(),
            specular: on_file.specular.into(),
            specular_exponent: on_file.specular_exponent.into(),
        })
    }
}
