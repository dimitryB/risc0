  // Retarget the in-tree evalfast EF_WIDE reduction to this RV32IM-only MSL
  // header. Four products accumulate exactly before one Montgomery reduction.
  static constexpr Fp dot4_reduce(uint32_t x0, uint32_t y0,
                                  uint32_t x1, uint32_t y1,
                                  uint32_t x2, uint32_t y2,
                                  uint32_t x3, uint32_t y3) {
    uint64_t t = uint64_t(x0) * y0 + uint64_t(x1) * y1 +
                 uint64_t(x2) * y2 + uint64_t(x3) * y3;
    uint32_t lo = uint32_t(t);
    uint32_t hi = uint32_t(t >> 32);
    uint32_t folded = hi - Fp::P;
    hi = hi < folded ? hi : folded;
    uint32_t m = lo * Fp::M;
    uint32_t r = hi - uint32_t((uint64_t(m) * Fp::P) >> 32);
    uint32_t plus = r + Fp::P;
    return Fp::fromRaw(r < plus ? r : plus);
  }

  constexpr FpExt operator*(FpExt rhs) const {
    const uint32_t a0 = elems[0].asRaw(), a1 = elems[1].asRaw();
    const uint32_t a2 = elems[2].asRaw(), a3 = elems[3].asRaw();
    const uint32_t b0 = rhs.elems[0].asRaw(), b1 = rhs.elems[1].asRaw();
    const uint32_t b2 = rhs.elems[2].asRaw(), b3 = rhs.elems[3].asRaw();
    const uint32_t b1n = (rhs.elems[1] * NBETA).asRaw();
    const uint32_t b2n = (rhs.elems[2] * NBETA).asRaw();
    const uint32_t b3n = (rhs.elems[3] * NBETA).asRaw();
    return FpExt(dot4_reduce(a0, b0, a1, b3n, a2, b2n, a3, b1n),
                 dot4_reduce(a0, b1, a1, b0, a2, b3n, a3, b2n),
                 dot4_reduce(a0, b2, a1, b1, a2, b0, a3, b3n),
                 dot4_reduce(a0, b3, a1, b2, a2, b1, a3, b0));
  }
