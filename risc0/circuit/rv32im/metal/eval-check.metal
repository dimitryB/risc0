
kernel void eval_check(device Fp* check,
                      const device Fp* ctrl,
                      const device Fp* data,
                      const device Fp* accum,
                      const device Fp* mix,
                      const device Fp* out,
                      const device FpExt* poly_mix,
                      const device Fp* normalization,
                      const device uint& po2,
                      const device uint& domain,
                      uint cycle [[thread_position_in_grid]]) {
    if (cycle >= domain) return;
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum, poly_mix);
    // The domain is 4 * steps: (3 * rou^cycle)^steps has period four.
    // Host-generated factors are checked against the CPU's original per-cycle
    // exponentiation for every output. Avoid that divergent power loop here.
    FpExt ret = tot * normalization[cycle & 3u];
    check[domain * 0 + cycle] = ret.elems[0];
    check[domain * 1 + cycle] = ret.elems[1];
    check[domain * 2 + cycle] = ret.elems[2];
    check[domain * 3 + cycle] = ret.elems[3];
}
