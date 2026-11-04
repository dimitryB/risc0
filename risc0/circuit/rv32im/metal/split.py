"""Split generated RV32IM chain functions at a small live-value boundary.

Only immutable trace loads and literal Fp constants are reissued in the tail.
The original statements and their order are otherwise preserved. The CPU/CUDA
body check runs before this pass; the Metal differential checks its output.
"""

import re


IDENT = re.compile(r"\bx\d+\b")
DEFINED = re.compile(r"^\s*(?:Fp|FpExt|auto)\s+(x\d+)\b")
LITERAL = re.compile(r"^\s*Fp (x\d+)\(\d+\);$")
INPUT = re.compile(r"^\s*Fp (x\d+) = (arg\d+)\[[^;]+\];$")


def split_chain(name, declaration, body):
    """Return a tail helper, replacement body, and a structural cut record."""
    lines = body.splitlines()
    assert len(lines) > 100 and name.startswith("rv32im_v2_")
    readonly = set(re.findall(r"\bconst device Fp\* (arg\d+)\b", declaration))
    definitions = {}
    uses = []
    for index, line in enumerate(lines):
        defined = DEFINED.match(line)
        if defined:
            assert defined[1] not in definitions, (name, defined[1])
            definitions[defined[1]] = index
        tokens = set(IDENT.findall(line))
        if defined:
            tokens.discard(defined[1])
        uses.append(tokens)
    # A cut must end at a full statement. Generated calls can span lines.
    depth = 0
    boundaries = set()
    for index, line in enumerate(lines):
        depth += line.count("(") - line.count(")")
        assert depth >= 0, (name, index)
        if depth == 0 and line.rstrip().endswith(";"):
            boundaries.add(index + 1)
    assert depth == 0, name
    candidates = []
    for cut in range(int(len(lines) * 0.45), int(len(lines) * 0.55)):
        if cut not in boundaries:
            continue
        used = set().union(*uses[cut:])
        live = {var for var in used if definitions.get(var, cut) < cut}
        literal = {var for var in live if LITERAL.match(lines[definitions[var]])}
        reload = {var for var in live if (m := INPUT.match(lines[definitions[var]])) and m[2] in readonly}
        passed = live - literal - reload
        candidates.append((len(passed), cut, sorted(passed), literal | reload))
    assert candidates, name
    _, cut, passed, reissued = min(candidates)
    # Reissue in original order, in case future generated loads depend on a
    # prior literal. Every current direct input load depends only on arguments.
    prelude = ["  uint32_t mask = size - 1;"]
    prelude += [lines[definitions[var]] for var in sorted(reissued, key=definitions.get)]
    helper_name = name + "_tail"
    params = ["idx", "size"] + list(dict.fromkeys(re.findall(r"\barg\d+\b", declaration))) + ["poly_mix"]
    formal = declaration.replace(" " + name + "(", " " + helper_name + "(", 1)
    assert formal != declaration and formal.endswith(")")
    formal = formal[:-1] + "".join(f", T{i} {var}" for i, var in enumerate(passed)) + ")"
    template = "template <" + ", ".join(f"typename T{i}" for i in range(len(passed))) + ">\n" if passed else ""
    helper = template + formal + " {\n" + "\n".join(prelude + lines[cut:]) + "\n}\n"
    call = "  return " + helper_name + "(" + ", ".join(params + passed) + ");"
    shortened = "\n".join(lines[:cut] + [call])
    record = {"function": name, "body_lines": len(lines), "cut_line": cut,
              "passed_values": passed, "reissued_values": sorted(reissued)}
    return helper, shortened, record
