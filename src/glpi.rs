use anyhow::{bail, Result};
use byteorder::ByteOrder;
use zerocopy::{FromBytes, U16};

use crate::{
    glf::{List, ListOnFile},
    order::Loader,
};

pub struct Armature {
    pub parents: Vec<u16>,
}

impl<'a> Loader<'a> for Armature {
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

        let mut parents = Vec::with_capacity(list.len());
        for v in list.iter() {
            let Some(v) = U16::<O>::read_from(v) else {
                bail!("Invalid list item size");
            };
            parents.push(v.get())
        }

        Ok(Armature { parents })
    }
}
