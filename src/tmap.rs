use std::mem;

use anyhow::{ensure, Result};
use byteorder::ByteOrder;
use zerocopy::{FromBytes, U16};

use crate::order::Loader;

#[derive(FromBytes)]
#[repr(C)]
pub struct TextureMapOnFile<O>
where
    O: ByteOrder,
{
    byte_order: U16<O>,
    _osk: U16<O>,
    stride: U16<O>,
    r#type: u8,
    flags: u8,
    base_x: U16<O>,
    base_y: U16<O>,
    width: U16<O>,
    height: U16<O>,
    origin_x: U16<O>,
    origin_y: U16<O>,
}

pub struct TextureMap {
    pub width: u16,
    pub height: u16,
    pub data: Vec<u8>,
}

impl<'a> Loader<'a> for TextureMap {
    type OnFile<O> = TextureMapOnFile<O>
    where
        O: ByteOrder;

    fn byte_order<O>(on_file: &Self::OnFile<O>) -> u16
    where
        O: ByteOrder,
    {
        on_file.byte_order.get()
    }

    fn into_native<O>(on_file: Self::OnFile<O>, full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder,
    {
        const BR_PMT_INDEX_8: u8 = 3;
        // Why'd it have to be this one?
        // All textures in the base game tmpls.3cn are indexed color.
        // At least it's a multiple of eight bits and it's not YUV.
        ensure!(
            on_file.r#type == BR_PMT_INDEX_8,
            "Unsupported texture format {}",
            on_file.r#type,
        );

        // I'm not sure what these are supposed to mean.
        // If they're always zero I don't need to find out.
        ensure!(
            on_file.base_x.get() == 0
                && on_file.base_y.get() == 0
                && on_file.origin_x.get() == 0
                && on_file.origin_y.get() == 0,
            "Unsupported texture origin",
        );

        assert_eq!(on_file.stride.get(), on_file.width.get());

        let mut data =
            Vec::with_capacity(on_file.width.get() as usize * on_file.height.get() as usize);
        let mut input = &full_input[mem::size_of::<TextureMapOnFile<O>>()..];
        for input_row in input.chunks_exact(on_file.stride.get() as usize) {
            data.extend_from_slice(&input_row[..on_file.width.get() as usize]);
            input = &input[on_file.stride.get() as usize - on_file.width.get() as usize..];
        }

        Ok(TextureMap {
            width: on_file.width.get(),
            height: on_file.height.get(),
            data,
        })
    }
}
