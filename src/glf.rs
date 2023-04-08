use std::{mem, ops::Index};

use anyhow::{bail, Result};
use byteorder::ByteOrder;
use zerocopy::{FromBytes, U16, U32};

use crate::order::Loader;

#[derive(Debug, FromBytes)]
#[repr(C)]
pub struct ListOnFile<O>
where
    O: ByteOrder,
{
    byte_order: U16<O>,
    _osk: U16<O>,
    entry_size: U32<O>,
    length: U32<O>,
}

pub struct List<'a> {
    data: &'a [u8],
    entry_size: u32,
}

impl<'a> List<'a> {
    pub fn from_file<O>(header: &ListOnFile<O>, full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder,
    {
        let Some(data) = full_input.get(mem::size_of::<ListOnFile<O>>()..mem::size_of::<ListOnFile<O>>()+header.entry_size.get() as usize * header.length.get() as usize) else {
            bail!("EOF in list");
        };

        Ok(List {
            data,
            entry_size: header.entry_size.get().max(1),
        })
    }

    pub fn get(&self, index: usize) -> Option<&[u8]> {
        self.data
            .get(self.entry_size as usize * index..self.entry_size as usize * (index + 1))
    }

    pub fn iter(&self) -> ListItems<'a> {
        self.into_iter()
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.entry_size as usize
    }
}

impl<'a> Index<usize> for List<'a> {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.entry_size as usize * index..self.entry_size as usize * (index + 1)]
    }
}

impl<'a> IntoIterator for &List<'a> {
    type Item = &'a [u8];

    type IntoIter = ListItems<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ListItems {
            data: self.data,
            entry_size: self.entry_size,
        }
    }
}

pub struct ListItems<'a> {
    data: &'a [u8],
    entry_size: u32,
}

impl<'a> Iterator for ListItems<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }

        let (first, rest) = self.data.split_at(self.entry_size as usize);
        self.data = rest;
        Some(first)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.data.len() / self.entry_size as usize;
        (size, Some(size))
    }
}

impl<'a> DoubleEndedIterator for ListItems<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }

        let (rest, last) = self
            .data
            .split_at(self.data.len() - self.entry_size as usize);
        self.data = rest;
        Some(last)
    }
}

impl<'a> Loader<'a> for List<'a> {
    type OnFile<O> = ListOnFile<O>
    where
        O: ByteOrder;

    fn byte_order<O>(on_file: &Self::OnFile<O>) -> u16
    where
        O: ByteOrder,
    {
        on_file.byte_order.get()
    }

    fn into_native<O>(header: Self::OnFile<O>, full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder,
    {
        List::from_file(&header, full_input)
    }
}
