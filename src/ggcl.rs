use std::mem;

use anyhow::{bail, Result};
use byteorder::ByteOrder;
use zerocopy::{FromBytes, U16, U32};

use crate::{
    brender::Scalar,
    ggf::{Group, GroupOnFile},
    order::Loader,
};

#[derive(FromBytes)]
#[repr(C)]
struct CelOnFile<O>
where
    O: ByteOrder,
{
    _sound_id: U32<O>,
    dwr: Scalar<O>,
}

pub struct Cell {
    pub dwr: f64,
    pub parts: Vec<CellPartSpec>,
}

#[derive(FromBytes)]
#[repr(C)]
struct CpsOnFile<O>
where
    O: ByteOrder,
{
    model_id: U16<O>,
    matrix_id: U16<O>,
}

#[derive(Debug)]
pub struct CellPartSpec {
    pub model_id: Option<u16>,
    pub matrix_id: u16,
}

pub struct AnimationCells {
    pub cells: Vec<Cell>,
}

impl<'a> Loader<'a> for AnimationCells {
    type OnFile<O> = GroupOnFile<O>
    where
        O: ByteOrder;

    fn byte_order<O>(on_file: &Self::OnFile<O>) -> u16
    where
        O: ByteOrder,
    {
        Group::byte_order(on_file)
    }

    fn into_native<O>(on_file: Self::OnFile<O>, full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder,
    {
        let group = Group::from_file(&on_file, full_input)?;

        let mut cells = Vec::with_capacity(group.len());
        for v in group.iter() {
            let Some(cel) = CelOnFile::<O>::read_from(v.fixed) else {
                bail!("Invalid fixed item size");
            };
            let mut parts = Vec::with_capacity(v.variable.len() / mem::size_of::<CpsOnFile<O>>());
            let mut cps_data = v.variable;
            while !cps_data.is_empty() {
                let Some(cps) = CpsOnFile::<O>::read_from_prefix(cps_data) else {
                    bail!("EOF in CPS");
                };
                cps_data = &cps_data[mem::size_of::<CpsOnFile<O>>()..];
                parts.push(CellPartSpec {
                    model_id: Some(cps.model_id.get()).filter(|v| *v != 65535),
                    matrix_id: cps.matrix_id.get(),
                });
            }
            cells.push(Cell {
                dwr: cel.dwr.into(),
                parts,
            });
        }

        Ok(AnimationCells { cells })
    }
}
