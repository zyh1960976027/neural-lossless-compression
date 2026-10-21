//! A batch C ABI over Constriction 0.4.2, with no floating point quantization.
//! Callers own all array memory; handles are exclusive and must not be shared.
use constriction::stream::{Decode, Encode};
use constriction::stream::model::{DecoderModel, EncoderModel, EntropyModel};
use constriction::stream::queue::{DefaultRangeDecoder, DefaultRangeEncoder};
use std::{borrow::Borrow, cell::Cell, num::NonZeroU32, panic::{catch_unwind, AssertUnwindSafe}, slice};

const TOTAL: u32 = 1 << 24;
struct Interval(u32, NonZeroU32);
impl EntropyModel<24> for Interval { type Symbol = (); type Probability = u32; }
impl EncoderModel<24> for Interval {
    fn left_cumulative_and_probability(&self, _: impl Borrow<()>) -> Option<(u32, NonZeroU32)> {
        Some((self.0, self.1))
    }
}
struct Cdf<'a> { ends: &'a [u32], invalid: Cell<bool> }
impl EntropyModel<24> for Cdf<'_> { type Symbol = u32; type Probability = u32; }
impl DecoderModel<24> for Cdf<'_> {
    fn quantile_function(&self, q: u32) -> (u32, u32, NonZeroU32) {
        let i = self.ends.partition_point(|&x| x <= q);
        if i == 0 || i >= self.ends.len() {
            self.invalid.set(true);
            return (0, 0, NonZeroU32::new(1).unwrap());
        }
        let (left, right) = (self.ends[i-1], self.ends[i]);
        if left > q || right <= q || right > TOTAL {
            self.invalid.set(true);
            return (0, 0, NonZeroU32::new(1).unwrap());
        }
        ((i-1) as u32, left, NonZeroU32::new(right-left).unwrap())
    }
}
fn guarded(f: impl FnOnce() -> Result<(), ()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) { Ok(Ok(())) => 0, _ => -1 }
}

#[no_mangle]
pub extern "C" fn ir_abi_version() -> u32 { 1 }
#[no_mangle]
pub extern "C" fn ir_encoder_new() -> *mut DefaultRangeEncoder {
    Box::into_raw(Box::new(DefaultRangeEncoder::new()))
}
#[no_mangle]
pub unsafe extern "C" fn ir_encoder_free(p: *mut DefaultRangeEncoder) {
    if !p.is_null() { drop(Box::from_raw(p)); }
}
#[no_mangle]
pub unsafe extern "C" fn ir_encode(p: *mut DefaultRangeEncoder, pairs: *const u32, count: usize) -> i32 {
    guarded(|| {
        let rows = slice::from_raw_parts(pairs, count.checked_mul(2).ok_or(())?);
        // Validate before mutating state. Each row contains left cumulative and frequency.
        if rows.chunks_exact(2).any(|r| r[1] == 0 || r[1] >= TOTAL || r[0] >= TOTAL || r[1] > TOTAL-r[0]) { return Err(()); }
        for r in rows.chunks_exact(2) {
            (*p).encode_symbol((), Interval(r[0], NonZeroU32::new(r[1]).unwrap())).map_err(|_| ())?;
        }
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn ir_compressed_size(p: *mut DefaultRangeEncoder) -> usize {
    (*p).get_compressed().len()
}
#[no_mangle]
pub unsafe extern "C" fn ir_compressed_copy(p: *mut DefaultRangeEncoder, out: *mut u32, size: usize) -> i32 {
    guarded(|| {
        let compressed = (*p).get_compressed();
        if compressed.len() != size { return Err(()); }
        slice::from_raw_parts_mut(out, size).copy_from_slice(&compressed);
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn ir_decoder_new(data: *const u32, size: usize) -> *mut DefaultRangeDecoder {
    let result = catch_unwind(|| DefaultRangeDecoder::from_compressed(slice::from_raw_parts(data, size).to_vec()));
    match result { Ok(Ok(d)) => Box::into_raw(Box::new(d)), _ => std::ptr::null_mut() }
}
#[no_mangle]
pub unsafe extern "C" fn ir_decoder_free(p: *mut DefaultRangeDecoder) {
    if !p.is_null() { drop(Box::from_raw(p)); }
}
#[no_mangle]
pub unsafe extern "C" fn ir_decode(p: *mut DefaultRangeDecoder, cdfs: *const u32, count: usize, width: usize, out: *mut u32) -> i32 {
    guarded(|| {
        if width < 3 || width > TOTAL as usize { return Err(()); }
        let rows = slice::from_raw_parts(cdfs, count.checked_mul(width).ok_or(())?);
        let symbols = slice::from_raw_parts_mut(out, count);
        for (cdf, symbol) in rows.chunks_exact(width).zip(symbols) {
            if cdf[0] != 0 || cdf[width-1] != TOTAL { return Err(()); }
            // Production CDFs are constructed monotonically on the GPU. Bounds
            // and the selected interval are checked here without an O(V) rescan.
            let model = Cdf { ends: cdf, invalid: Cell::new(false) };
            *symbol = (*p).decode_symbol(&model).map_err(|_| ())?;
            if model.invalid.get() { return Err(()); }
        }
        Ok(())
    })
}
