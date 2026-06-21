"""One-off: inspect binary path diversity in a binskel jsonl."""
import json
import sys
from collections import Counter


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else r"E:\ABCD_datasets\AB_train.jsonl"
    after_ds = Counter()
    path_prefix3 = Counter()
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            b = (d.get("binary") or "").replace("\\", "/")
            parts = [p for p in b.split("/") if p]
            if "datasets" in parts:
                i = parts.index("datasets")
                if i + 1 < len(parts):
                    after_ds[parts[i + 1]] += 1
                if i + 3 <= len(parts):
                    path_prefix3["/".join(parts[i : i + 3])] += 1
    print("path:", path)
    print("lines:", n)
    print("--- count: folder right after datasets/ ---")
    for k, v in after_ds.most_common():
        print(v, k)
    print("--- top path prefixes (datasets/X/Y) ---")
    for k, v in path_prefix3.most_common(30):
        print(v, k)


if __name__ == "__main__":
    main()
