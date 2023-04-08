use std::io::Read;

use anyhow::{bail, ensure, Result};
use bitvec::{field::BitField, prelude::*};
use byteorder::{BigEndian, ReadBytesExt};

pub fn decode(mut input: &[u8]) -> Result<Vec<u8>> {
    let len = input.read_u32::<BigEndian>()? as usize;
    let Some(input) = input.get(1..) else {
        bail!("Empty encoded data");
    };
    let mut input = input.view_bits::<Lsb0>();
    let mut output = Vec::with_capacity(len);

    loop {
        let length = match read_length(&mut input) {
            Some(Length::Ok(length)) => length,
            Some(Length::Break) => break,
            None => bail!("Unexpected EOF"),
        };

        let Some((bit, rest)) = input.split_first() else {
            bail!("Unexpected EOF");
        };
        input = rest;

        let destination = output.len();
        if !bit {
            ensure!(destination + length <= len, "Overflow");
            output.reserve(destination + length);

            let Some(bits) = input.get(..length * 8) else {
                bail!("Unexpected EOF");
            };
            input = &input[length * 8..];

            let (head, mut body, tail) = bits.bit_domain().region().unwrap();
            body.read_to_end(&mut output)?;
            if !tail.is_empty() || !head.is_empty() {
                output.push(
                    head.iter()
                        .chain(tail)
                        .fold(0, |a, b| (a >> 1) + (u8::from(*b) << 7)),
                );
            }

            continue;
        }

        let Some((offset, length)) = read_offset_length(&mut input, length) else {
            bail!("Invalid backref");
        };

        let Some(source) = output
            .len()
            .checked_sub(offset) else {
                eprintln!("{:02x?}", output);
            bail!("Offset out of range ({offset} > {})", output.len());
        };
        let destination = output.len();
        ensure!(
            destination + length <= len,
            "Overflow ({destination} + {length} > {len})",
        );
        for i in source..source + length {
            output.push(output[i]);
        }
    }

    Ok(output)
}

enum Length {
    Ok(usize),
    Break,
}

fn read_length<T, O>(input: &mut &BitSlice<T, O>) -> Option<Length>
where
    T: BitStore,
    O: BitOrder,
    BitSlice<T, O>: BitField,
{
    let max = 12.min(input.len());
    let length_length = input[..max].leading_ones();
    if length_length == max {
        return Some(Length::Break);
    }
    *input = &input[length_length + 1..];

    Some(Length::Ok(if length_length == 0 {
        1
    } else {
        let bits = input.get(..length_length)?;
        *input = &input[length_length..];
        bits.load_le::<u16>() as usize + (1 << length_length)
    }))
}

fn read_offset_length<T, O>(
    input: &mut &BitSlice<T, O>,
    base_length: usize,
) -> Option<(usize, usize)>
where
    T: BitStore,
    O: BitOrder,
    BitSlice<T, O>: BitField,
{
    let mut iter = input.iter();
    let (offset_length, offset_offset, length_offset) = if !*iter.next()? {
        (6, 0x0001, 1)
    } else if !*iter.next()? {
        (9, 0x0041, 1)
    } else if !*iter.next()? {
        (12, 0x0241, 1)
    } else {
        (20, 0x1241, 2)
    };
    *input = iter.as_bitslice();
    let bits = input.get(..offset_length)?;
    *input = &input[offset_length..];

    let offset = bits.load_le::<u32>() as usize + offset_offset;

    Some((offset, base_length + length_offset))
}
