# Fix compile_curl.sh: line 129 had echo without closing "; line 97 may break parser
from pathlib import Path

p = Path(__file__).resolve().parent / "compile_curl.sh"
raw = p.read_bytes().splitlines()

# 0-based indices
if len(raw) >= 129:
    # Line 129: unclosed echo
    if b"curl_o${opt}.ll" in raw[128] and not raw[128].strip().endswith(b'"'):
        raw[128] = b'    echo "    WARN: curl_o${opt}.ll not generated (see clang errors above)."'

if len(raw) >= 97:
    # Line 97: long warn - replace if looks like garbled warn (contains curl_o0.ll in echo)
    L = raw[96]
    if b"curl_o0.ll" in L and L.strip().startswith(b'  echo "'):
        raw[96] = b'  echo "    WARN: no .bc files; skip curl_o0.ll (O0 DWARF-only). Check clang/HAVE_CONFIG_H above."'

p.write_bytes(b"\n".join(raw) + b"\n")
print("OK", p)
