use std::{marker::PhantomData, mem};

use anyhow::{bail, Result};
use byteorder::ByteOrder;
use zerocopy::{FromBytes, U16, U32};

#[derive(Debug, FromBytes)]
#[repr(C)]
pub struct GroupOnFile<O>
where
    O: ByteOrder,
{
    byte_order: U16<O>,
    _osk: U16<O>,
    length_entries: U32<O>,
    data_length_bytes: U32<O>,
    _cloc_free: U32<O>,
    fixed: U32<O>,
}

pub struct Group<'a, O>
where
    O: ByteOrder,
{
    fixed: usize,
    entries: &'a [u8],
    data: &'a [u8],
    _phantom: PhantomData<O>,
}

#[derive(FromBytes)]
#[repr(C)]
struct Loc<O>
where
    O: ByteOrder,
{
    offset: U32<O>,
    length: U32<O>,
}

impl<'a, O> Group<'a, O>
where
    O: ByteOrder,
{
    pub fn from_file(header: &GroupOnFile<O>, full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder,
    {
        let Some(remainder) = full_input.get(mem::size_of::<GroupOnFile<O>>()..mem::size_of::<GroupOnFile<O>>()+mem::size_of::<Loc<O>>()*header.length_entries.get() as usize+header.data_length_bytes.get() as usize) else {
            bail!("EOF in group");
        };

        let (data, entries) = remainder.split_at(header.data_length_bytes.get() as usize);

        Ok(Group {
            fixed: header.fixed.get() as usize,
            entries,
            data,
            _phantom: PhantomData,
        })
    }

    pub fn get(&self, index: usize) -> Option<GroupEntry> {
        let loc = Loc::<O>::read_from(
            &self.entries[index * mem::size_of::<Loc<O>>()..(index + 1) * mem::size_of::<Loc<O>>()],
        )?;
        let data = &self.data
            [loc.offset.get() as usize..loc.offset.get() as usize + loc.length.get() as usize];
        let (fixed, variable) = data.split_at(self.fixed);

        Some(GroupEntry { fixed, variable })
    }

    pub fn iter(&self) -> GroupItems<'a, O> {
        self.into_iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len() / mem::size_of::<Loc<O>>()
    }

    pub fn byte_order(on_file: &GroupOnFile<O>) -> u16
    where
        O: ByteOrder,
    {
        on_file.byte_order.get()
    }
}

impl<'a, O> IntoIterator for &Group<'a, O>
where
    O: ByteOrder,
{
    type Item = GroupEntry<'a>;

    type IntoIter = GroupItems<'a, O>;

    fn into_iter(self) -> Self::IntoIter {
        GroupItems {
            fixed: self.fixed,
            entries: self.entries,
            data: self.data,
            _phantom: PhantomData,
        }
    }
}

pub struct GroupItems<'a, O>
where
    O: ByteOrder,
{
    fixed: usize,
    entries: &'a [u8],
    data: &'a [u8],
    _phantom: PhantomData<O>,
}

#[derive(Debug)]
pub struct GroupEntry<'a> {
    pub fixed: &'a [u8],
    pub variable: &'a [u8],
}

impl<'a, O> Iterator for GroupItems<'a, O>
where
    O: ByteOrder,
{
    type Item = GroupEntry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.entries.is_empty() {
            return None;
        }

        let (first, rest) = self.entries.split_at(mem::size_of::<Loc<O>>());
        let loc = Loc::<O>::read_from(first).unwrap();
        self.entries = rest;

        let data = &self.data
            [loc.offset.get() as usize..loc.offset.get() as usize + loc.length.get() as usize];
        let (fixed, variable) = data.split_at(self.fixed);

        Some(GroupEntry { fixed, variable })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.entries.len() / mem::size_of::<Loc<O>>();
        (size, Some(size))
    }
}

impl<'a, O> DoubleEndedIterator for GroupItems<'a, O>
where
    O: ByteOrder,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.entries.is_empty() {
            return None;
        }

        let (rest, last) = self
            .entries
            .split_at(self.entries.len() - mem::size_of::<Loc<O>>());
        let loc = Loc::<O>::read_from(last).unwrap();
        self.entries = rest;

        let data = &self.data
            [loc.offset.get() as usize..loc.offset.get() as usize + loc.length.get() as usize];
        let (fixed, variable) = data.split_at(self.fixed);

        Some(GroupEntry { fixed, variable })
    }
}
