use byteorder::ByteOrder;
use zerocopy::{FromBytes, I16, I32, U16};

#[derive(Clone, Copy, FromBytes)]
#[repr(transparent)]
pub struct Scalar<O>(I32<O>)
// signed 15.16
where
    O: ByteOrder;

impl<O> Scalar<O>
where
    O: ByteOrder,
{
    pub fn is_zero(&self) -> bool {
        self.0.get() == 0
    }
}

impl<O> From<Scalar<O>> for f64
where
    O: ByteOrder,
{
    fn from(value: Scalar<O>) -> Self {
        value.0.get() as f64 / 65536.0
    }
}

#[derive(Clone, Copy, FromBytes)]
#[repr(transparent)]
pub struct Fraction<O>(I16<O>)
// signed 0.15
where
    O: ByteOrder;

impl<O> From<Fraction<O>> for f32
where
    O: ByteOrder,
{
    fn from(value: Fraction<O>) -> Self {
        value.0.get() as f32 / 32768.0
    }
}

#[derive(Clone, Copy, FromBytes)]
#[repr(transparent)]
pub struct UFraction<O>(U16<O>)
// unsigned 0.16
where
    O: ByteOrder;

impl<O> From<UFraction<O>> for f32
where
    O: ByteOrder,
{
    fn from(value: UFraction<O>) -> Self {
        value.0.get() as f32 / 65536.0
    }
}
