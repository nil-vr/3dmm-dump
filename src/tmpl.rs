use anyhow::Result;
use byteorder::ByteOrder;
use zerocopy::{FromBytes, U16, U32};

use crate::{brender::UFraction, order::Loader};

#[derive(FromBytes)]
#[repr(C)]
pub struct TemplateOnFile<O>
where
    O: ByteOrder,
{
    pub byte_order: U16<O>,
    _osk: U16<O>,
    pub xa_rest: UFraction<O>,
    pub ya_rest: UFraction<O>,
    pub za_rest: UFraction<O>,
    _pad: U16<O>,
    pub grftmpl: U32<O>,
}

#[derive(Debug)]
pub struct Template {
    pub xa_rest: f32,
    pub ya_rest: f32,
    pub za_rest: f32,
}

impl<'a> Loader<'a> for Template {
    type OnFile<O> = TemplateOnFile<O>
    where
        O: ByteOrder;

    fn byte_order<O>(on_file: &Self::OnFile<O>) -> u16
    where
        O: ByteOrder,
    {
        on_file.byte_order.get()
    }

    fn into_native<O>(data: Self::OnFile<O>, _full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder,
    {
        Ok(Template {
            xa_rest: data.xa_rest.into(),
            ya_rest: data.ya_rest.into(),
            za_rest: data.za_rest.into(),
        })
    }
}
