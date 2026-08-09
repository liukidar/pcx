"""Mutation testing: break the library on purpose, check the suite notices.

A test that passes both before and after a deliberate bug is not testing
anything. Every mutation below is a genuine behaviour change — no equivalent
mutants (e.g. `0.5*(u-h)**2` is exactly `0.5*(h-u)**2`, so flipping that
subtraction proves nothing and is deliberately not listed).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTATIONS = [
    (
        "se_energy: prediction error becomes a sum instead of a difference",
        "pcx/predictive_coding/_energy.py",
        'e = vode.get("h") - vode.get("u")',
        'e = vode.get("h") + vode.get("u")',
    ),
    (
        "se_energy: drop the 1/2 factor",
        "pcx/predictive_coding/_energy.py",
        "return 0.5 * (e * e)",
        "return e * e",
    ),
    (
        "ce_energy: drop the negation (sign flip on cross-entropy)",
        "pcx/predictive_coding/_energy.py",
        "return -(vode.get",
        "return (vode.get",
    ),
    (
        "Optim: skip applying updates",
        "pcx/utils/_optim.py",
        "if apply_updates:\n            self.apply_updates(module, updates)",
        "if False:\n            self.apply_updates(module, updates)",
    ),
    (
        "Optim: negate the update (gradient ascent)",
        "pcx/utils/_optim.py",
        "lambda u, p: set(p, eqx.apply_updates(get(p), get(u)))",
        "lambda u, p: set(p, eqx.apply_updates(get(p), -get(u)))",
    ),
    (
        "tree_ref: never detect a duplicate (silently breaks parameter sharing)",
        "pcx/core/_tree.py",
        "elif isinstance(x, BaseParam) and ((_ref := _seen(id(x))) is not None):",
        "elif isinstance(x, BaseParam) and ((_ref := _seen(id(x))) is not None) and False:",
    ),
    (
        "Param.__iadd__: return a new array instead of mutating in place",
        "pcx/core/_parameter.py",
        "    def __iadd__(self, __other):\n"
        "        self._value = self._value.__add__(get(__other))\n"
        "        return self",
        "    def __iadd__(self, __other):\n        return self._value.__add__(get(__other))",
    ),
    (
        "RandomKeyGenerator: stop advancing the key (every draw repeats)",
        "pcx/core/_random.py",
        "self.set(values[0])",
        "pass",
    ),
    (
        "Module.train/eval: make eval() a no-op",
        "pcx/core/_module.py",
        "def eval(self",
        "def eval_unused(self",
    ),
    (
        "save_params: write zeros instead of the real weights",
        "pcx/utils/_serialisation.py",
        "_data[jtu.keystr(key)] = param.get()",
        "_data[jtu.keystr(key)] = param.get() * 0",
    ),
]


def run_suite() -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    first_failure = ""
    for line in out.splitlines():
        if line.startswith("FAILED") or line.startswith("ERROR"):
            first_failure = line.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "")
            break
    return r.returncode != 0, first_failure


def main() -> int:
    results = []
    for name, relpath, old, new in MUTATIONS:
        path = ROOT / relpath
        original = path.read_text(encoding="utf-8")
        if old not in original:
            results.append((name, "SKIP", "pattern not found"))
            continue
        try:
            path.write_text(original.replace(old, new, 1), encoding="utf-8")
            caught, detail = run_suite()
            results.append((name, "caught" if caught else "SURVIVED", detail))
        finally:
            path.write_text(original, encoding="utf-8")

    width = max(len(n) for n, _, _ in results)
    print()
    for name, verdict, detail in results:
        mark = {"caught": "  caught", "SURVIVED": "SURVIVED", "SKIP": "    skip"}[verdict]
        print(f"{mark}  {name:<{width}}  {detail}")

    survived = [n for n, v, _ in results if v == "SURVIVED"]
    applied = [r for r in results if r[1] != "SKIP"]
    print(f"\n{len(applied) - len(survived)}/{len(applied)} mutations caught")
    if survived:
        print("Survivors (blind spots):")
        for n in survived:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
