from pathlib import Path
p = Path(__file__).resolve().parent / "compile_curl.sh"
lines = p.read_bytes().splitlines()
lines[96] = (
    b'  echo "    WARN: no .bc files; skip curl_o0.ll (O0 DWARF-only). '
    b'Check clang/HAVE_CONFIG_H above."'
)
p.write_bytes(b"\n".join(lines) + b"\n")
print("line 97 fixed")
