use std::{fs::File, io::BufWriter, mem, path::Path};

use anyhow::{bail, ensure, Result};
use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use png::{BitDepth, ColorType, Encoder};
use zerocopy::{FromBytes, U16, U32};

use crate::order::{BYTE_ORDER_NATIVE, BYTE_ORDER_SWAPPED};

#[derive(FromBytes)]
#[repr(C)]
struct Rect<O>
where
    O: ByteOrder,
{
    left: U32<O>,
    top: U32<O>,
    right: U32<O>,
    bottom: U32<O>,
}

#[derive(FromBytes)]
#[repr(C)]
struct BitmapHeader<O>
where
    O: ByteOrder,
{
    byte_order: U16<O>,
    _osk: U16<O>,
    mask: u8,
    fill: u8,
    reserved: U16<O>,
    rc: Rect<O>,
    size: U32<O>,
}

pub fn convert_mbmp_to_png(full_input: &[u8], output_path: &Path) -> Result<()> {
    fn convert<O>(header: &BitmapHeader<O>, full_input: &[u8], output_path: &Path) -> Result<()>
    where
        O: ByteOrder,
    {
        ensure!(
            header.byte_order.get() == BYTE_ORDER_NATIVE,
            "Wrong signature {}",
            header.byte_order.get()
        );
        ensure!(header.size.get() as usize == full_input.len(), "Wrong size");

        let mut input = &full_input[mem::size_of::<BitmapHeader<O>>()..];

        let mut row_lengths =
            Vec::with_capacity(header.rc.bottom.get() as usize - header.rc.top.get() as usize);
        for _ in 0..header.rc.bottom.get() - header.rc.top.get() {
            row_lengths.push(input.read_u16::<O>()?);
        }

        let mut image = vec![0u8; header.rc.right.get() as usize * header.rc.bottom.get() as usize];
        if header.rc.right.get() > header.rc.left.get() {
            let dst_row_range = header.rc.left.get() as usize..header.rc.right.get() as usize;
            for (row_length, mut dst_row) in row_lengths.iter().map(|l| *l as usize).zip(
                image
                    .chunks_mut(header.rc.right.get() as usize)
                    .skip(header.rc.top.get() as usize),
            ) {
                dst_row = &mut dst_row[dst_row_range.clone()];
                ensure!(
                    row_length <= input.len(),
                    "Source row contains too many bytes"
                );
                let (mut src_row, rest) = input.split_at(row_length);
                input = rest;
                while let Ok(transparent) = src_row.read_u8() {
                    dst_row = &mut dst_row[transparent as usize..];
                    let opaque = src_row.read_u8()? as usize;
                    ensure!(
                        opaque <= dst_row.len(),
                        "Source row contains too many pixels"
                    );
                    let (chunk, rest) = dst_row.split_at_mut(opaque);
                    dst_row = rest;
                    if header.mask == 1 {
                        chunk.fill(header.fill);
                    } else {
                        ensure!(opaque <= src_row.len(), "Truncated row");
                        let (in_chunk, rest) = src_row.split_at(opaque);
                        src_row = rest;
                        chunk.copy_from_slice(in_chunk);
                    }
                }
            }
        }

        if !image.is_empty() {
            let writer = BufWriter::new(File::create(output_path)?);
            let mut encoder = Encoder::new(writer, header.rc.right.get(), header.rc.bottom.get());
            encoder.set_color(ColorType::Grayscale);
            encoder.set_depth(BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(&image)?;
            writer.finish()?;
        }

        Ok(())
    }

    let Some(header) = BitmapHeader::<LittleEndian>::read_from_prefix(full_input) else {
        bail!("Bitmap too small");
    };

    if header.byte_order.get() == BYTE_ORDER_SWAPPED {
        let header = BitmapHeader::<BigEndian>::read_from_prefix(full_input).unwrap();
        convert(&header, full_input, output_path)
    } else {
        convert(&header, full_input, output_path)
    }
}
