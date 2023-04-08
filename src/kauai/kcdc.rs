use anyhow::{bail, ensure, Result};
use bitvec::{field::BitField, prelude::*};
use byteorder::{BigEndian, ReadBytesExt};

const OFFSET_STOP: usize = 0x101240;

pub fn decode(mut input: &[u8]) -> Result<Vec<u8>> {
    let len = input.read_u32::<BigEndian>()? as usize;
    let Some(input) = input.get(1..) else {
        bail!("Empty encoded data");
    };
    let mut input = input.view_bits::<Lsb0>();
    let mut output = Vec::with_capacity(len);

    loop {
        let Some((bit, rest)) = input.split_first() else {
            bail!("Unexpected EOF");
        };
        input = rest;

        if !bit {
            ensure!(output.len() < len, "Overflow");
            let Some(bits) = input.get(0..8) else {
                bail!("EOF in byte literal");
            };
            input = &input[8..];
            output.push(bits.load_le::<u8>());
            continue;
        }

        let Some((offset, length)) = read_offset_length(&mut input) else {
            bail!("Invalid backref");
        };

        if offset == OFFSET_STOP {
            break;
        }

        let Some(source) = output
            .len()
            .checked_sub(offset) else {
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

fn read_offset_length<T, O>(input: &mut &BitSlice<T, O>) -> Option<(usize, usize)>
where
    T: BitStore,
    O: BitOrder,
    BitSlice<T, O>: BitField,
    BitVec<<T as bitvec::store::BitStore>::Unalias, O>: BitField,
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
    if offset == OFFSET_STOP {
        return Some((offset, 0));
    }

    let max = 12.min(input.len());
    let length_length = input[..max].leading_ones();
    if length_length == max {
        return None;
    }
    *input = &input[length_length + 1..];

    let length = if length_length == 0 {
        length_offset + 1
    } else {
        let bits = input.get(..length_length)?;
        *input = &input[length_length..];
        bits.load_le::<u16>() as usize + (1 << length_length) + length_offset
    };

    Some((offset, length))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unusual_byte_groupings)]
    fn decode_aba() {
        assert_eq!(
            &b"aba"[..],
            &decode(&[
                0x0,
                0x0,
                0x0,
                0x3,
                0x0,
                0b0_1000011u8.reverse_bits(),
                0b0_0_010001u8.reverse_bits(),
                0b10_0_10000u8.reverse_bits(),
                0b110_1111_1u8.reverse_bits(),
                0b11111111u8.reverse_bits(),
                0b11111111u8.reverse_bits(),
                0b111_11111u8.reverse_bits(),
            ])
            .unwrap(),
        );
    }

    #[test]
    fn read_offset_length_1_36() {
        // The documentation in codkauai.cpp has the bits reversed.
        let mut bits = bits![u32, Lsb0;
            // 6 bit offset
            0,
            // offset = 0 + 1 = 1
            0, 0, 0, 0, 0, 0,
            // length encoded in next 5 bits
            1, 1, 1, 1, 1, 0,
            // length = 1 + 3 + (1 << 5) = 36
            1, 1, 0, 0, 0,
        ];
        assert_eq!(Some((1, 36)), read_offset_length(&mut bits));
        assert!(bits.is_empty());
    }

    #[test]
    fn read_offset_length_3_4() {
        // The documentation in codkauai.cpp has the bits reversed.
        let mut bits = bits![u32, Lsb0;
            // 6 bit offset
            0,
            // offset = 2 + 1 = 3
            0, 1, 0, 0, 0, 0,
            // length encoded in next 1 bit
            1, 0,
            // length = 1 + 1 + (1 << 1) = 4
            1,
        ];
        assert_eq!(Some((3, 4)), read_offset_length(&mut bits));
        assert!(bits.is_empty());
    }

    #[test]
    fn read_offset_length_4_5() {
        // The documentation in codkauai.cpp has the bits reversed.
        let mut bits = bits![u32, Lsb0;
            // 6 bit offset
            0,
            // offset = 3 + 1 = 4
            1, 1, 0, 0, 0, 0,
            // length encoded in next 2 bits
            1, 1, 0,
            // length = 1 + 0 + (1 << 2) = 5
            0, 0,
        ];
        assert_eq!(Some((4, 5)), read_offset_length(&mut bits));
        assert!(bits.is_empty());
    }

    #[test]
    fn read_offset_length_5_2() {
        // The documentation in codkauai.cpp has the bits reversed.
        let mut bits = bits![u32, Lsb0;
            // 6 bit offset
            0,
            // offset = 4 + 1 = 5
            0, 0, 1, 0, 0, 0,
            // length encoded in next 0 bits
            0,
            // length = 1 + 0 + (1 << 0) = 2
        ];
        assert_eq!(Some((5, 2)), read_offset_length(&mut bits));
        assert!(bits.is_empty());
    }

    #[test]
    fn read_offset_length_stop() {
        let mut bits = bits![u32, Lsb0;
            // 20 bit offset
            1, 1, 1,
            // offset = 1048575 + 4673 = 1053248
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];
        assert_eq!(Some((OFFSET_STOP, 0)), read_offset_length(&mut bits));
        assert!(bits.is_empty());
    }
}
