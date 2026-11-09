"""Build the upload zip from bot/ and check the competition's limits.

Tomorrow prefer the kit's own ``poker-agent pack bot dist/bot.zip`` (it runs
the server's checks). This script is a local stand-in with the same rules:
  * agent.py at the zip root, hidden files and __pycache__ left out
  * no file imports poker_engine or pokerkit
  * <= 50 MB zipped, <= 200 MB unpacked, <= 1000 entries, agent.py <= 1 MB
  * imports limited to stdlib + numpy/scipy/pandas/sklearn/torch/lightgbm/
    xgboost/onnxruntime/numba/eval7 + the SDK packages (warned, not enforced)

Run: python3 tools/pack.py  ->  dist/bot.zip
"""
import ast
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "bot")
OUT = os.path.join(ROOT, "dist", "bot.zip")
ALLOWED = {"numpy", "scipy", "pandas", "sklearn", "torch", "lightgbm", "xgboost", "onnxruntime",
           "numba", "eval7", "poker_agent_sdk", "poker_agent_protocol", "pb"}
FORBIDDEN = {"poker_engine", "pokerkit"}


def files():
    for d, dirs, fs in os.walk(SRC):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x != "__pycache__")
        for f in sorted(fs):
            if f.startswith(".") or f.endswith(".pyc"):
                continue
            p = os.path.join(d, f)
            if os.path.islink(p):
                raise SystemExit(f"symlinked file not allowed: {p}")
            yield p


def check_imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    bad, unknown = [], []
    std = set(sys.stdlib_module_names)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for n in names:
            top = n.split(".")[0]
            if top in FORBIDDEN:
                bad.append(n)
            elif top not in std and top not in ALLOWED:
                unknown.append(n)
    return bad, unknown


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fs = list(files())
    unpacked = 0
    for p in fs:
        unpacked += os.path.getsize(p)
        if p.endswith(".py"):
            bad, unknown = check_imports(p)
            if bad:
                raise SystemExit(f"FORBIDDEN import in {p}: {bad}")
            if unknown:
                print(f"warning: {os.path.relpath(p, ROOT)} imports {unknown} (not in the allowed list)")
    agent = os.path.join(SRC, "agent.py")
    if not os.path.exists(agent):
        raise SystemExit("bot/agent.py missing")
    if os.path.getsize(agent) > 1_000_000:
        raise SystemExit("agent.py over 1 MB")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in fs:
            info = zipfile.ZipInfo(os.path.relpath(p, SRC), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(p, "rb") as fh:
                z.writestr(info, fh.read())
    size = os.path.getsize(OUT)
    with zipfile.ZipFile(OUT) as z:
        entries = len(z.namelist())
    assert entries <= 1000 and size <= 50 * 2**20 and unpacked <= 200 * 2**20
    print(f"wrote {OUT}: {entries} files, {size / 1024:.0f} KB zipped, {unpacked / 1024:.0f} KB unpacked")


if __name__ == "__main__":
    main()
