// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

/// \file
/// Defines FpExt, a finite field F_p^4, based on Fp via the irreducible polynomial x^4 - 11.

#include "ef_fp.h"

namespace risc0 {

// Defines instead of constexpr to appease CUDAs limitations around constants.
// undef'd at the end of this file.
#define BETA Fp(11)
#define NBETA Fp(Fp::P - 11)

/// Instances of FpExt are element of a finite field F_p^4.  They are represented as elements of
/// F_p[X] / (X^4 - 11). Basically, this is a 'big' finite field (about 2^128 elements), which is
/// used when the security of various operations depends on the size of the field.  It has the field
/// Fp as a subfield, which means operations by the two are compatible, which is important.  The
/// irreducible polynomial was chosen to be the simplest possible one, x^4 - B, where 11 is the
/// smallest B which makes the polynomial irreducible.
struct FpExt {
  /// The elements of FpExt, elems[0] + elems[1]*X + elems[2]*X^2 + elems[3]*x^4
  Fp elems[4];

  /// Default constructor makes the zero elements
  constexpr FpExt() {}

  /// Initialize from uint32_t
  explicit constexpr FpExt(uint32_t x) {
    elems[0] = x;
    elems[1] = 0;
    elems[2] = 0;
    elems[3] = 0;
  }

  /// Convert from Fp to FpExt.
  explicit constexpr FpExt(Fp x) {
    elems[0] = x;
    elems[1] = 0;
    elems[2] = 0;
    elems[3] = 0;
  }

  /// Explicitly construct an FpExt from parts
  constexpr FpExt(Fp a, Fp b, Fp c, Fp d) {
    elems[0] = a;
    elems[1] = b;
    elems[2] = c;
    elems[3] = d;
  }

  /// Get an 'invalid' FpExt value
  static constexpr inline FpExt invalid() {
    return FpExt(Fp::invalid(), Fp::invalid(), Fp::invalid(), Fp::invalid());
  }

  // Implement the addition/subtraction overloads
  constexpr FpExt operator+=(FpExt rhs) {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] += rhs.elems[i];
    }
    return *this;
  }

  constexpr FpExt operator-=(FpExt rhs) {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] -= rhs.elems[i];
    }
    return *this;
  }

  constexpr FpExt operator+(FpExt rhs) const {
    FpExt result = *this;
    result += rhs;
    return result;
  }

  constexpr FpExt operator-(FpExt rhs) const {
    FpExt result = *this;
    result -= rhs;
    return result;
  }

  constexpr FpExt operator-() const { return FpExt() - *this; }

  // Implement the simple multiplication case by the subfield Fp
  // Fp * FpExt is done as a free function due to C++'s operator overloading rules.
  constexpr FpExt operator*=(Fp rhs) {
    for (uint32_t i = 0; i < 4; i++) {
      elems[i] *= rhs;
    }
    return *this;
  }

  constexpr FpExt operator*(Fp rhs) const {
    FpExt result = *this;
    result *= rhs;
    return result;
  }

  constexpr FpExt operator+=(Fp rhs) {
    elems[0] += rhs;
    return *this;
  }

  constexpr FpExt operator+(Fp rhs) const {
    FpExt result = *this;
    result += rhs;
    return result;
  }

  constexpr FpExt operator-=(Fp rhs) {
    elems[0] -= rhs;
    return *this;
  }

  constexpr FpExt operator-(Fp rhs) const {
    FpExt result = *this;
    result -= rhs;
    return result;
  }

  // Now we get to the interesting case of multiplication.  Basically, multiply out the polynomial
  // representations, and then reduce module x^4 - B, which means powers >= 4 get shifted back 4 and
  // multiplied by -beta.  We could write this as a double loops with some if's and hope it gets
  // unrolled properly, but it's small enough to just hand write.
// EF_EXT_LAZY on ROCm too (eval_check_ef_hip.cu): 219 vs 226 ms per po2 20 eval_check on
// gfx1100, 265 vs 271 ms on gfx1201.
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && defined(EF_EXT_LAZY)
  // Inlined: as an out-of-line call (IGC's default for this size) every call
  // passed operands through private memory and drained the pipeline; that
  // calling convention, not the arithmetic, dominated the Intel eval_check.
  // nvcc spells it __forceinline__: with cl.exe as its host compiler it has no
  // __attribute__ (and it inlines this anyway).
#ifdef __NVCC__
  __forceinline__
#else
  __attribute__((always_inline))
#endif
#endif
  constexpr FpExt operator*(FpExt rhs) const {
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && defined(EF_EXT_LAZY)
    // Lazy reduction: fold NBETA into b (3 Montgomery muls), then each output
    // coefficient is a 4-term dot product of raw Montgomery values summed
    // exactly in 64 bits (as 32-bit lo/hi words with explicit carries; IGC
    // emits add-with-carry) and reduced once. Bounds: each product < P^2, so
    // a sum < 4P^2 < 2^64 and hi < 4P^2/2^32 < 1.875P; one conditional fold
    // makes hi < P, so the subtraction-form REDC lands in (-P, P).
    const uint32_t a0 = elems[0].asRaw(), a1 = elems[1].asRaw();
    const uint32_t a2 = elems[2].asRaw(), a3 = elems[3].asRaw();
    const uint32_t b0 = rhs.elems[0].asRaw(), b1 = rhs.elems[1].asRaw();
    const uint32_t b2 = rhs.elems[2].asRaw(), b3 = rhs.elems[3].asRaw();
    const uint32_t b1n = (rhs.elems[1] * NBETA).asRaw();
    const uint32_t b2n = (rhs.elems[2] * NBETA).asRaw();
    const uint32_t b3n = (rhs.elems[3] * NBETA).asRaw();
    auto mulhi = [](uint32_t x, uint32_t y) { return uint32_t((uint64_t(x) * uint64_t(y)) >> 32); };
#if defined(RISC0_INTEL_EVAL_LOHI)
    // Reduce straight from the lo/hi words (no uint64 pack/unpack). Fastest on
    // the restructured eval_check (evalfast/ pipeline); only that build
    // defines RISC0_INTEL_EVAL_LOHI.
#if defined(EF_WIDE)
    // 64-bit accumulate (IMAD.WIDE.U32): sum < 4P^2 < 2^64, identical bits.
    auto redc4 = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                     uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3) -> Fp {
      uint64_t t = uint64_t(x0) * y0 + uint64_t(x1) * y1 + uint64_t(x2) * y2 + uint64_t(x3) * y3;
      uint32_t lo = uint32_t(t), hi = uint32_t(t >> 32);
      uint32_t hf = hi - Fp::P;
      hi = hi < hf ? hi : hf;
      uint32_t r = hi - mulhi(lo * Fp::M, Fp::P);
      uint32_t u = r + Fp::P;
      return Fp::fromRaw(r < u ? r : u);
    };
#else
    auto redc4 = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                     uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3) -> Fp {
      uint32_t lo = x0 * y0, hi = mulhi(x0, y0);
      uint32_t p = x1 * y1, n = lo + p;
      hi += mulhi(x1, y1) + (n < p ? 1u : 0u);
      lo = n;
      p = x2 * y2;
      n = lo + p;
      hi += mulhi(x2, y2) + (n < p ? 1u : 0u);
      lo = n;
      p = x3 * y3;
      n = lo + p;
      hi += mulhi(x3, y3) + (n < p ? 1u : 0u);
      lo = n;
      uint32_t hf = hi - Fp::P;
      hi = hi < hf ? hi : hf;
      uint32_t r = hi - mulhi(lo * Fp::M, Fp::P);
      uint32_t t = r + Fp::P;
      return Fp::fromRaw(r < t ? r : t);
    };
#endif
    return FpExt(redc4(a0, b0, a1, b3n, a2, b2n, a3, b1n),
                 redc4(a0, b1, a1, b0, a2, b3n, a3, b2n),
                 redc4(a0, b2, a1, b1, a2, b0, a3, b3n),
                 redc4(a0, b3, a1, b2, a2, b1, a3, b0));
#else
    // dot4 returns the exact 64-bit sum packed into a uint64_t, and redc
    // unpacks it: this is the measured-fastest form (v13). Reducing straight
    // from the lo/hi words issues fewer instructions but measured ~10 ms slower.
    auto dot4 = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                    uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3) -> uint64_t {
      uint32_t lo = x0 * y0, hi = mulhi(x0, y0);
      uint32_t p = x1 * y1, n = lo + p;
      hi += mulhi(x1, y1) + (n < p ? 1u : 0u);
      lo = n;
      p = x2 * y2;
      n = lo + p;
      hi += mulhi(x2, y2) + (n < p ? 1u : 0u);
      lo = n;
      p = x3 * y3;
      n = lo + p;
      hi += mulhi(x3, y3) + (n < p ? 1u : 0u);
      lo = n;
      return (uint64_t(hi) << 32) | lo;
    };
    auto redc = [&](uint64_t s) -> Fp {
      uint32_t lo = uint32_t(s), hi = uint32_t(s >> 32);
      uint32_t hf = hi - Fp::P;
      hi = hi < hf ? hi : hf;
      uint32_t r = hi - mulhi(lo * Fp::M, Fp::P);
      uint32_t t = r + Fp::P;
      return Fp::fromRaw(r < t ? r : t);
    };
    return FpExt(redc(dot4(a0, b0, a1, b3n, a2, b2n, a3, b1n)),
                 redc(dot4(a0, b1, a1, b0, a2, b3n, a3, b2n)),
                 redc(dot4(a0, b2, a1, b1, a2, b0, a3, b3n)),
                 redc(dot4(a0, b3, a1, b2, a2, b1, a3, b0)));
#endif
#else
    // Rename the element arrays to something small for readability
#define a elems
#define b rhs.elems
    return FpExt(a[0] * b[0] + NBETA * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]),
                 a[0] * b[1] + a[1] * b[0] + NBETA * (a[2] * b[3] + a[3] * b[2]),
                 a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + NBETA * (a[3] * b[3]),
                 a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0]);
#undef a
#undef b
#endif
  }
  constexpr FpExt operator*=(FpExt rhs) {
    *this = *this * rhs;
    return *this;
  }

  // Equality
  constexpr bool operator==(FpExt rhs) const {
    for (uint32_t i = 0; i < 4; i++) {
      if (elems[i] != rhs.elems[i]) {
        return false;
      }
    }
    return true;
  }

  constexpr bool operator!=(FpExt rhs) const { return !(*this == rhs); }

  constexpr Fp constPart() const { return elems[0]; }
};

/// Overload for case where LHS is Fp (RHS case is handled as a method)
constexpr inline FpExt operator*(Fp a, FpExt b) {
  return b * a;
}

// Commutate the two arguments
constexpr inline FpExt operator+(Fp a, FpExt b) {
  return b + a;
}

// Promote a to FpExt, then add.
constexpr inline FpExt operator-(Fp a, FpExt b) {
  return a + (-b);
}

/// Raise an FpExt to a power
constexpr inline FpExt pow(FpExt x, size_t n) {
  FpExt tot(1);
  while (n != 0) {
    if (n % 2 == 1) {
      tot *= x;
    }
    n = n / 2;
    x *= x;
  }
  return tot;
}

/// Compute the multiplicative inverse of an FpExt.
constexpr inline FpExt inv(FpExt in) {
#define a in.elems
  // Compute the multiplicative inverse by basically looking at FpExt as a composite field and using
  // the same basic methods used to invert complex numbers.  We imagine that initially we have a
  // numerator of 1, and a denominator of a. i.e out = 1 / a; We set a' to be a with the first and
  // third components negated.  We then multiply the numerator and the denominator by a', producing
  // out = a' / (a * a'). By construction (a * a') has 0's in it's first and third elements.  We
  // call this number, 'b' and compute it as follows.
  Fp b0 = a[0] * a[0] + BETA * (a[1] * (a[3] + a[3]) - a[2] * a[2]);
  Fp b2 = a[0] * (a[2] + a[2]) - a[1] * a[1] + BETA * (a[3] * a[3]);
  // Now, we make b' by inverting b2.  When we multiply both sizes by b', we get out = (a' * b') /
  // (b * b').  But by construction b * b' is in fact an element of Fp, call it c.
  Fp c = b0 * b0 + BETA * b2 * b2;
  // But we can now invert C directly, and multiply by a'*b', out = a'*b'*inv(c)
  Fp ic = inv(c);
  // Note: if c == 0 (really should only happen if in == 0), our 'safe' version of inverse results
  // in ic == 0, and thus out = 0, so we have the same 'safe' behavior for FpExt.  Oh, and since we
  // want to multiply everything by ic, it's slightly faster to premultiply the two parts of b by ic
  // (2 multiplies instead of 4)
  b0 *= ic;
  b2 *= ic;
  return FpExt(a[0] * b0 + BETA * a[2] * b2,
               -a[1] * b0 + NBETA * a[3] * b2,
               -a[0] * b2 + a[2] * b0,
               a[1] * b2 - a[3] * b0);
#undef a
}

#undef BETA
#undef NBETA

} // namespace risc0
