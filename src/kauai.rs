use std::borrow::Cow;

use anyhow::{bail, Result};

mod kcd2;
mod kcdc;

pub fn unpack(input: &[u8]) -> Result<Cow<'_, [u8]>> {
    let Some(packed) = input.get(0..4) else {
        bail!("Too short");
    };

    Ok(match packed {
        b"puak" => Cow::Borrowed(&input[4..]),
        b"apak" => Cow::Owned(decode(&input[4..])?),
        _ => bail!("Unsupported signature {}", packed.escape_ascii()),
    })
}

pub fn decode(input: &[u8]) -> Result<Vec<u8>> {
    let Some(codec) = input.get(..4) else {
        bail!("Too short");
    };
    Ok(match codec {
        b"KCDC" => kcdc::decode(&input[4..])?,
        b"KCD2" => kcd2::decode(&input[4..])?,
        _ => bail!("Unsupported codec {}", codec.escape_ascii()),
    })
}
