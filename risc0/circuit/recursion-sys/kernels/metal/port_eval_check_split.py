#!/usr/bin/env python3
"""Port the pinned recursion gated-sum split to six ordered MSL kernels."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data.encode()).hexdigest()


def extract_parts(source):
    marker = "namespace risc0::circuit::recursion::cuda {\n"
    assert source.count(marker) == 1
    assert "constexpr int kNumParts = 6;" in source
    parts = []
    for part in range(6):
        signature = (
            f"__device__ __forceinline__ FpExt poly_fp_part{part}(uint32_t idx, "
            "uint32_t size, const Fp* ctrl, const Fp* out, const Fp* data, "
            "const Fp* mix, const Fp* accum) {"
        )
        assert source.count(signature) == 1, f"recursion part {part} changed"
        start = source.index(signature) + len(signature)
        end = source.index("\n}\n", start)
        body = source[start:end]
        assert "return " in body and "poly_mix[" in body
        assert not any(token in body for token in
                       ("__device__", "__HIP", "REC_", "reinterpret_cast", "std::"))
        parts.append(body)
    return parts


def port(source):
    parts = extract_parts(source)
    output = [
        "// Generated from recursion eval_check_split.py's pre-lazy gated-sum output.\n",
        "// The six kernels keep the CUDA split's part order and sum regrouping.\n",
        '#include <metal_stdlib>\n#include "fp.h"\n#include "fpext.h"\n',
        "using namespace metal;\nconstant size_t INV_RATE = 4;\n",
    ]
    for part, body in enumerate(parts):
        output.append(
            f"inline FpExt poly_fp_part{part}(uint32_t idx, uint32_t size, "
            "const device Fp* ctrl, const device Fp* out, "
            "const device Fp* data, const device Fp* mix, "
            "const device Fp* accum, const device FpExt* poly_mix) {"
            + body + "\n}\n"
        )
    for part in range(6):
        output.append(
            f"kernel void eval_check_part{part}(device Fp* check, "
            "const device Fp* ctrl, const device Fp* data, "
            "const device Fp* accum, const device Fp* mix, "
            "const device Fp* out, const device FpExt* poly_mix, "
            "const device Fp& rou, const device uint32_t& po2, "
            "const device uint32_t& domain, "
            "uint cycle [[thread_position_in_grid]]) {\n"
            "  if (cycle >= domain) return;\n"
            f"  FpExt tot = poly_fp_part{part}(cycle, domain, ctrl, out, "
            "data, mix, accum, poly_mix);\n"
        )
        if part:
            output.append(
                "  tot = FpExt(check[cycle], check[domain + cycle], "
                "check[2 * domain + cycle], check[3 * domain + cycle]) + tot;\n"
            )
        if part == 5:
            output.append(
                "  Fp x = pow(rou, cycle);\n"
                "  Fp y = pow(Fp(3) * x, 1u << po2);\n"
                "  tot = tot * inv(y - Fp(1));\n"
            )
        for component in range(4):
            output.append(
                f"  check[{component} * domain + cycle] = "
                f"tot.elems[{component}];\n"
            )
        output.append("}\n")
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_text()
    result = port(source)
    args.output.write_text(result)
    args.output.with_suffix(".json").write_text(json.dumps({
        "schema_version": 1,
        "input_sha256": sha(source),
        "output_sha256": sha(result),
        "port_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "part_count": 6,
        "cuda_split": "gated sums before lazy-REDC and HIP-specific variants",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
