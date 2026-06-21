"""One-off: replace MD_EXTRA block with source _binskel_md_extra.sh"""
from pathlib import Path

OLD = """NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
MD_EXTRA=(--workers "$WORKERS" --num-opcodes "$NUM_OPCODES" --src-vocab-size "$SRC_VOCAB")"""

NEW = """NUM_OPCODES="${NUM_OPCODES:-1024}"
SRC_VOCAB="${SRC_VOCAB:-32000}"
WORKERS="${WORKERS:-8}"
# shellcheck source=/dev/null
source "$WORKDIR/scripts/_binskel_md_extra.sh"
"""

ROOT = Path(__file__).resolve().parent
for p in list(ROOT.glob("build*.sh")) + list((ROOT / "os_only").glob("build*.sh")):
    t = p.read_text(encoding="utf-8")
    if OLD not in t:
        print("SKIP:", p)
        continue
    p.write_text(t.replace(OLD, NEW, 1), encoding="utf-8")
    print("OK:", p)
