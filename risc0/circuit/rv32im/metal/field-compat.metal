// The retained Metal FpExt header makes Fp promotion explicit. The generated
// RV32IM polynomial uses the mixed addition/subtraction supported by CUDA.
// Keep those promotions here, without rewriting generated arithmetic.
constexpr inline FpExt operator+(Fp a, FpExt b) { return FpExt(a) + b; }
constexpr inline FpExt operator-(Fp a, FpExt b) { return FpExt(a) - b; }
constexpr inline FpExt operator+(FpExt a, Fp b) { return a + FpExt(b); }
constexpr inline FpExt operator-(FpExt a, Fp b) { return a - FpExt(b); }
