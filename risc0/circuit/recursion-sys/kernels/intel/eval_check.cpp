// Intel SYCL eval_check kernel for recursion circuit.
// This file is appended to an amalgamation that already contains:
//   - fp.h, fpext.h includes
//   - The poly_fp function definition (from poly_fp.cpp)
//   - namespace risc0::circuit::recursion with kInvRate defined
// So we only need the SYCL kernel wrapper here.

#include <sycl/sycl.hpp>
#include <cstring>

using namespace risc0;

static const char* make_error(const char* msg) { return strdup(msg); }

extern "C" const char* risc0_circuit_recursion_intel_eval_check(
    void* queue_ptr,
    void* d_check,
    const void* d_ctrl,
    const void* d_data,
    const void* d_accum,
    const void* d_mix,
    const void* d_out,
    const void* d_poly_mix,
    uint32_t rou_raw,
    uint32_t po2,
    uint32_t domain)
{
    auto* q = static_cast<sycl::queue*>(queue_ptr);

    try {
        auto* check = static_cast<Fp*>(d_check);
        auto* ctrl = static_cast<Fp*>(const_cast<void*>(d_ctrl));
        auto* data = static_cast<Fp*>(const_cast<void*>(d_data));
        auto* accum = static_cast<Fp*>(const_cast<void*>(d_accum));
        auto* mix = static_cast<Fp*>(const_cast<void*>(d_mix));
        auto* out = static_cast<Fp*>(const_cast<void*>(d_out));
        auto* poly_mix = static_cast<FpExt*>(const_cast<void*>(d_poly_mix));

        // Construct rou from raw Montgomery bits
        Fp rou_val;
        std::memcpy(&rou_val, &rou_raw, sizeof(uint32_t));
#ifdef RECURSION_QTABLE
        // y = (3*rou^cycle)^(2^po2) depends only on cycle mod 4 (rou has order
        // 2^(po2+2)), so the vanishing quotient takes 4 values: compute on host.
        // Guard: the table is exact only if rou^(4 * 2^po2) == 1 (the Rust
        // caller passes ROU_FWD[po2 + 2], whose order is exactly 2^(po2+2)).
        if (pow(rou_val, size_t(4) << po2) != Fp(1)) {
            return make_error("recursion eval_check: rou order does not divide 2^(po2+2); quotient table invalid");
        }
        Fp qt0, qt1, qt2, qt3;
        {
          Fp qv[4];
          for (uint32_t k = 0; k < 4; k++) {
            Fp x = Fp(3) * pow(rou_val, k);
            Fp y = pow(x, size_t(1) << po2);
            qv[k] = inv(y - Fp(1));
          }
          qt0 = qv[0]; qt1 = qv[1]; qt2 = qv[2]; qt3 = qv[3];
        }
#endif

        // Default WG=512: the kernel is built with 256 GRF/thread (-Os,
        // RISC0_INTEL_EVAL_FAST, DisableRecompilation=1), which caps the
        // work-group at 512. Measured flat for 128..512 (41-42 ms/proof on a B70
        // when tuned; 169-172 ms on a B580 with the large private size below).
        // Runtime-tunable via RISC0_RECURSION_EVAL_CHECK_WG for sweeps.
        #ifndef RECURSION_EVAL_WG
#define RECURSION_EVAL_WG 512
#endif
        uint32_t WG_SIZE = RECURSION_EVAL_WG;
        if (const char* s = std::getenv("RISC0_RECURSION_EVAL_CHECK_WG")) {
            int v = std::atoi(s);
            if (v == 16 || v == 32 || v == 64 || v == 128 || v == 256 || v == 512 || v == 1024) {
                WG_SIZE = (uint32_t)v;
            }
        }
        uint32_t global_size = ((domain + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
        const uint32_t stride = global_size;

        // A loop over the work item's rows (one row: the grid covers the domain), not an early
        // return: with this form and build.rs's reload distance (`remat_distance`), IGC keeps
        // poly_fp's values out of private memory (896 B per work item instead of 527,232 B), and
        // eval_check takes 68 ms per recursion proof on an Arc B580 instead of 172. Either change
        // alone left the large private size. build.rs warns if it comes back.
        q->parallel_for(
            sycl::nd_range<1>(global_size, WG_SIZE),
            [=](sycl::nd_item<1> item) {
              for (uint32_t cycle = item.get_global_id(0); cycle < domain; cycle += stride) {

                // args order: ctrl(0), out/global(1), data(2), mix(3), accum(4)
                Fp* args[5] = {ctrl, out, data, mix, accum};

                FpExt tot = circuit::recursion::poly_fp(
                    (size_t)cycle, (size_t)domain, poly_mix, args);

#ifdef RECURSION_QTABLE
                uint32_t k = cycle & 3;
                Fp quot = k == 0 ? qt0 : (k == 1 ? qt1 : (k == 2 ? qt2 : qt3));
#else
                Fp x = Fp(3) * pow(rou_val, cycle);
                Fp y = pow(x, uint32_t(1) << po2);
                Fp quot = inv(y - Fp(1));
#endif

                for (uint32_t i = 0; i < 4; i++) {
                    check[i * domain + cycle] = tot.elems[i] * quot;
                }
              }
            });
        q->wait();

        return nullptr;
    } catch (const sycl::exception& e) {
        return make_error(e.what());
    } catch (const std::exception& e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("Unknown error in intel eval_check");
    }
}
