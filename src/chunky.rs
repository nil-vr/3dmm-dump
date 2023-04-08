use std::{
    borrow::Cow, cmp::Ordering, collections::HashMap, ffi::OsString, fmt, mem,
    os::windows::prelude::OsStringExt,
};

use anyhow::{bail, ensure, Context, Result};
use bitflags::bitflags;
use byteorder::{ByteOrder, NativeEndian, ReadBytesExt};
use zerocopy::{FromBytes, U16, U32};

use crate::{kauai, order::Loader};

const CURRENT_VERSION: u16 = 5;
const MINIMUM_VERSION: u16 = 1;

#[derive(Debug, FromBytes)]
#[repr(C)]
struct DataVersion<O>
where
    O: ByteOrder,
{
    current: U16<O>,
    backwards_compatible: U16<O>,
}

impl<O> DataVersion<O>
where
    O: ByteOrder,
{
    fn is_readable(&self, current_version: u16, minimum_version: u16) -> bool {
        self.current.get() >= minimum_version && self.backwards_compatible.get() <= current_version
    }
}

#[derive(Debug, FromBytes)]
#[repr(C)]
pub struct Prefix<O>
where
    O: ByteOrder,
{
    magic: U32<O>,
    creator: U32<O>,
    version: DataVersion<O>,
    byte_order: U16<O>,
    _osk: U16<O>,
    eof: U32<O>,
    index_offset: U32<O>,
    index_len: U32<O>,
    free_map: U32<O>,
    reserved: [u8; 23],
}

#[derive(Debug, FromBytes)]
#[repr(C)]
struct GroupOnFile<O>
where
    O: ByteOrder,
{
    byte_order: U16<O>,
    _osk: U16<O>,
    iloc_mac: U32<O>,
    bv_mac: U32<O>,
    cloc_free: U32<O>,
    cb_fixed: U32<O>,
}

#[derive(Debug, FromBytes)]
#[repr(C)]
struct Loc<O>
where
    O: ByteOrder,
{
    offset: U32<O>,
    length: U32<O>,
}

#[derive(Clone, Copy, Eq, FromBytes, Hash, PartialEq)]
#[repr(transparent)]
pub struct ChunkTag<O = NativeEndian>(U32<O>)
where
    O: ByteOrder;

impl<O> fmt::Debug for ChunkTag<O>
where
    O: ByteOrder,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.0.get().to_be_bytes();
        write!(
            f,
            "\"{}{}{}{}\"",
            (bytes[0] as char).escape_debug(),
            (bytes[1] as char).escape_debug(),
            (bytes[2] as char).escape_debug(),
            (bytes[3] as char).escape_debug(),
        )
    }
}

impl<O> PartialEq<&str> for ChunkTag<O>
where
    O: ByteOrder,
{
    fn eq(&self, other: &&str) -> bool {
        self == other.as_bytes()
    }
}

impl<O> PartialEq<[u8]> for ChunkTag<O>
where
    O: ByteOrder,
{
    fn eq(&self, other: &[u8]) -> bool {
        self.0.get().to_be_bytes() == other
    }
}

#[derive(Clone, Copy, Debug, Eq, FromBytes, Hash, PartialEq)]
#[repr(C)]
pub struct ChunkId<O = NativeEndian>
where
    O: ByteOrder,
{
    pub tag: ChunkTag<O>,
    pub number: U32<O>,
}

impl Ord for ChunkId<NativeEndian> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tag
            .0
            .get()
            .cmp(&other.tag.0.get())
            .then_with(|| self.number.get().cmp(&other.number.get()))
    }
}

impl PartialOrd for ChunkId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<O> ChunkId<O>
where
    O: ByteOrder,
{
    fn swap<O2>(&self) -> ChunkId<O2>
    where
        O2: ByteOrder,
    {
        ChunkId {
            tag: ChunkTag(U32::new(self.tag.0.get())),
            number: U32::new(self.number.get()),
        }
    }
}

#[derive(FromBytes)]
#[repr(transparent)]
struct GrfcrpCb<O>
where
    O: ByteOrder,
{
    inner: U32<O>,
}

impl<O> GrfcrpCb<O>
where
    O: ByteOrder,
{
    pub fn grfcrp(&self) -> u8 {
        (self.inner.get() & 0xff) as u8
    }

    pub fn length(&self) -> u32 {
        self.inner.get() >> 8
    }
}

impl<O> fmt::Debug for GrfcrpCb<O>
where
    O: ByteOrder,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GrfcrpCb")
            .field("grfcrp", &self.grfcrp())
            .field("length", &self.length())
            .finish()
    }
}

#[derive(Debug, FromBytes)]
#[repr(C)]
struct ChunkRepresentationSmall<O>
where
    O: ByteOrder,
{
    id: ChunkId<O>,
    offset: U32<O>,
    grfcrp_cb: GrfcrpCb<O>,
    child_count: U16<O>,
    owner_count: U16<O>,
}

#[derive(Debug, FromBytes)]
#[repr(C)]
struct ChildChunkRef<O>
where
    O: ByteOrder,
{
    chunk_id: ChunkId<O>,
    child_id: U32<O>,
}

#[derive(Debug, FromBytes)]
#[repr(C)]
struct StringHeader<O>
where
    O: ByteOrder,
{
    osk: U16<O>,
    len: u8,
}

bitflags! {
    #[derive(Debug)]
    pub struct ChunkFlags: u8 {
        const EXTRA = 0x01;
        const LONER = 0x02;
        const PACKED = 0x04;
        const MARK_T = 0x08;
        const FOREST = 0x10;
    }
}

#[derive(Debug)]
pub struct IndexEntry<'a> {
    pub offset: u32,
    pub flags: ChunkFlags,
    pub length: u32,
    pub name: Cow<'a, str>,
    pub children: Vec<ChildLink>,
}

impl<'a> IndexEntry<'a> {
    pub fn get_child(&self, child_id: u32, tag: impl AsRef<[u8]>) -> Option<&ChunkId> {
        let tag: [u8; 4] = tag.as_ref().try_into().ok()?;
        let tag = u32::from_be_bytes(tag);
        let index = self
            .children
            .binary_search_by(|c| {
                c.child_id
                    .cmp(&child_id)
                    .then_with(|| c.chunk_id.tag.0.get().cmp(&tag))
            })
            .ok()?;
        Some(&self.children[index].chunk_id)
    }
}

#[derive(Debug)]
pub struct ChildLink {
    pub chunk_id: ChunkId,
    pub child_id: u32,
}

pub struct ChunkyFile<'a> {
    pub data: &'a [u8],
    pub index: HashMap<ChunkId, IndexEntry<'a>>,
}

impl<'a> ChunkyFile<'a> {
    pub fn get_chunk(&self, entry: &IndexEntry) -> Result<Cow<'a, [u8]>> {
        let data = &self.data[entry.offset as usize..entry.offset as usize + entry.length as usize];
        if entry.flags.contains(ChunkFlags::PACKED) {
            Ok(Cow::Owned(
                kauai::decode(data).with_context(|| format!("Unpacking {entry:?}"))?,
            ))
        } else {
            Ok(Cow::Borrowed(data))
        }
    }
}

impl<'a> Loader<'a> for ChunkyFile<'a> {
    type OnFile<O> = Prefix<O> where O: ByteOrder;

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
        ensure!(
            &header.magic.get().to_le_bytes() == b"CHN2",
            "Wrong magic {}",
            header.magic.get().to_le_bytes().escape_ascii(),
        );
        ensure!(
            header.version.is_readable(CURRENT_VERSION, MINIMUM_VERSION),
            "Incompatible version",
        );

        let Some(index) = full_input.get(header.index_offset.get() as usize..header.index_offset.get() as usize + header.index_len.get() as usize) else {
            bail!("Index out of bounds");
        };

        let index = Group::load(index)?.0;

        Ok(ChunkyFile {
            data: full_input,
            index,
        })
    }
}

struct Group<'a>(HashMap<ChunkId, IndexEntry<'a>>);

impl<'a> Loader<'a> for Group<'a> {
    type OnFile<O> = GroupOnFile<O> where O: ByteOrder;

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
        let mut index = HashMap::new();

        let input = &full_input[mem::size_of::<GroupOnFile<O>>()..];
        for entry in input[header.bv_mac.get() as usize..]
            .chunks_exact(mem::size_of::<Loc<O>>())
            .take(header.iloc_mac.get() as usize)
        {
            let Some(loc) = Loc::<O>::read_from(entry) else {
                bail!("EOF in index entries");
            };

            let mut data = &input
                [loc.offset.get() as usize..loc.offset.get() as usize + loc.length.get() as usize];
            let Some(representation) = ChunkRepresentationSmall::<O>::read_from_prefix(data) else {
                bail!("EOF in chunk representation");
            };

            data = &data[mem::size_of::<ChunkRepresentationSmall<O>>()..];
            let mut children = Vec::with_capacity(representation.child_count.get() as usize);
            for _ in 0..representation.child_count.get() {
                let Some(child) = ChildChunkRef::<O>::read_from_prefix(data) else {
                    bail!("EOF in chunk children");
                };
                children.push(ChildLink {
                    chunk_id: child.chunk_id.swap(),
                    child_id: child.child_id.get(),
                });
                data = &data[mem::size_of::<ChildChunkRef<O>>()..];
            }

            children.sort_unstable_by(|a, b| {
                a.child_id
                    .cmp(&b.child_id)
                    .then_with(|| a.chunk_id.cmp(&b.chunk_id))
            });

            let name = if !data.is_empty() {
                let Some(string) = StringHeader::<O>::read_from_prefix(data) else {
                    bail!("EOF in chunk name");
                };
                data = &data[mem::size_of::<StringHeader<O>>()..];
                match string.osk.get() {
                    0x0303 => Cow::Borrowed(std::str::from_utf8(&data[..string.len as usize])?),
                    0x0505 => {
                        let mut value = Vec::with_capacity(string.len as usize);
                        for c in data
                            .chunks_exact(2)
                            .take(string.len as usize)
                            .map(|mut c| c.read_u16::<O>().unwrap())
                        {
                            value.push(c);
                        }
                        let Ok(value) = OsString::from_wide(&value).into_string() else {
                            bail!("Invalid utf-16");
                        };
                        Cow::Owned(value)
                    }
                    _ => {
                        bail!("Unsupported string encoding");
                    }
                }
            } else {
                Cow::Borrowed("")
            };

            index.insert(
                representation.id.swap(),
                IndexEntry {
                    offset: representation.offset.get(),
                    flags: ChunkFlags::from_bits_retain(representation.grfcrp_cb.grfcrp()),
                    length: representation.grfcrp_cb.length(),
                    name,
                    children,
                },
            );
        }

        Ok(Group(index))
    }
}
