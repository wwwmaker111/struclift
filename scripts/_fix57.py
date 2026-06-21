from pathlib import Path
p = Path(__file__).resolve().parent / "compile_curl.sh"
lines = p.read_bytes().splitlines()
lines[56] = (
    b'[ -f "$CURL_REAL" ] && cp -f "$CURL_REAL" curl_o0 || '
    b'{ echo "ERROR: curl binary not found"; exit 1; }'
)
p.write_bytes(b"\n".join(lines) + b"\n")
print("line 57 fixed")
