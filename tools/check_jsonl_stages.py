"""One-off: validate jsonl lines for stage requirements."""
import json
import sys
from pathlib import Path

from struclift.data.binskel_jsonl import _validate_sample


def main() -> None:
    path = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    lines = path.read_text(encoding="utf-8").splitlines()[:n]
    for i, line in enumerate(lines):
        d = json.loads(line)
        fn = d.get("func_name", "?")
        s1 = _validate_sample(d, require_source=True)
        s2 = _validate_sample(d, require_alignment=True)
        s3 = _validate_sample(d, require_sft=True)
        has_sft = all(k in d for k in ("sft_input_ids", "sft_labels", "slot_types"))
        print(
            f"{i + 1}. {fn}: "
            f"stage1={s1 or 'OK'} | stage2={s2 or 'OK'} | "
            f"stage3_sft={s3 or 'OK'} | has_sft_keys={has_sft}",
        )


if __name__ == "__main__":
    main()
