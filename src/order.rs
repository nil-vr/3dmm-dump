use anyhow::{bail, Result};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use zerocopy::FromBytes;

pub const BYTE_ORDER_NATIVE: u16 = 0x0001;
pub const BYTE_ORDER_SWAPPED: u16 = 0x0100;

pub trait Loader<'a>: 'a + Sized {
    type OnFile<O>: FromBytes
    where
        O: ByteOrder;

    fn byte_order<O>(on_file: &Self::OnFile<O>) -> u16
    where
        O: ByteOrder;

    fn into_native<O>(on_file: Self::OnFile<O>, full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder;

    fn load(full_input: &'a [u8]) -> Result<Self> {
        let Some(on_disk) = Self::OnFile::<LittleEndian>::read_from_prefix(full_input) else {
            bail!("{} input too small", std::any::type_name::<Self>())
        };
        match Self::byte_order(&on_disk) {
            BYTE_ORDER_NATIVE => Self::into_native(on_disk, full_input),
            BYTE_ORDER_SWAPPED => {
                let Some(on_disk) = Self::OnFile::<BigEndian>::read_from_prefix(full_input) else {
                    bail!("{} input too small", std::any::type_name::<Self>())
                };
                Self::into_native(on_disk, full_input)
            }
            other => bail!(
                "Unexpected byte order {:04x} in {}",
                other,
                std::any::type_name::<Self>(),
            ),
        }
    }
}
