//! IEEE 754 binary16 (half-precision) bit cast — no `half` crate dep.
//!
//! Used by `macos::matvec_f32_impl` to cast f32 weights to f16 once at
//! call time (production caches the cast result on `MetalState`).
//! Kept in its own module so the conversion can be unit-tested on
//! Linux even though the kernel encoding lives in `macos.rs`.

/// Convert an `f32` to its IEEE 754 binary16 bit pattern.
///
/// Round-to-nearest-even, NaN and infinity preserved, subnormals flushed
/// when the source exponent is below -14. Matches `half::f16::from_f32`
/// to within ulp-1 on normal values; sign/inf/nan/zero are exact.
#[allow(dead_code)]
pub(crate) fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        let m16 = if mant != 0 {
            0x200 | (mant >> 13) as u16
        } else {
            0
        };
        return sign | 0x7c00 | m16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let mant = mant | 0x800000;
        let shift = 14 - new_exp;
        let m16 = (mant >> shift) as u16;
        let round_bit = (mant >> (shift - 1)) & 1;
        return sign | (m16 + round_bit as u16);
    }
    let m16 = (mant >> 13) as u16;
    let round_bit = (mant >> 12) & 1;
    let sticky = (mant & 0xfff) != 0;
    let mut packed = sign | ((new_exp as u16) << 10) | m16;
    if round_bit == 1 && (sticky || (m16 & 1) == 1) {
        packed += 1;
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_round_trips() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    }

    #[test]
    fn one_is_exact() {
        // 1.0 in f16 = 0x3c00
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-1.0), 0xbc00);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
    }

    #[test]
    fn small_normals_match_known_values() {
        // 0.5 = 0x3800, 0.25 = 0x3400, 1.5 = 0x3e00
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(0.25), 0x3400);
        assert_eq!(f32_to_f16_bits(1.5), 0x3e00);
    }

    #[test]
    fn overflow_to_inf() {
        // f16 max ≈ 65504; 1e6 overflows
        let bits = f32_to_f16_bits(1.0e6);
        assert_eq!(bits, 0x7c00);
        let nbits = f32_to_f16_bits(-1.0e6);
        assert_eq!(nbits, 0xfc00);
    }

    #[test]
    fn nan_stays_nan() {
        let bits = f32_to_f16_bits(f32::NAN);
        // exp field is all-ones, mantissa nonzero
        assert_eq!(bits & 0x7c00, 0x7c00);
        assert_ne!(bits & 0x03ff, 0);
    }

    #[test]
    fn inf_preserved() {
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
    }
}
