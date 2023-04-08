use anyhow::{bail, Result};
use byteorder::{ByteOrder, ReadBytesExt};
use zerocopy::{FromBytes, U32};

use crate::{
    ggf::{Group, GroupOnFile},
    order::Loader,
};

pub struct Costumes {
    pub part_sets: Vec<Vec<u32>>,
}

impl<'a> Loader<'a> for Costumes {
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

        let mut part_sets = Vec::with_capacity(group.len());
        for mut v in group.iter() {
            let Some(entries) = U32::<O>::read_from(v.fixed) else {
                bail!("Invalid fixed item size");
            };
            let mut set_materials = Vec::with_capacity(entries.get() as usize);
            for _ in 0..entries.get() {
                set_materials.push(v.variable.read_u32::<O>()?);
            }
            part_sets.push(set_materials);
        }

        Ok(Costumes { part_sets })
    }
}
