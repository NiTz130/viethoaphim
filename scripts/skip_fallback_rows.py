"""Set status='skip' on rows whose text_vi starts with '[CHƯA DỊCH]' in review.csv."""
import csv
from pathlib import Path

csv_path = Path(r"C:\code\viethoaphimv4\jobs\Tập 1-9\translation\review.csv")
text = csv_path.read_text(encoding="utf-8-sig")

# Detect the actual delimiter
dialect = csv.Sniffer().sniff(text[:4096])
delimiter = dialect.delimiter

rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
header = rows[0]
status_idx = header.index("status")
text_vi_idx = header.index("text_vi")
sid_idx = header.index("segment_id")

skipped = []
for row in rows[1:]:
    if row[text_vi_idx].startswith("[CHƯA DỊCH]"):
        row[status_idx] = "skip"
        skipped.append(row[sid_idx])

# Write back as UTF-8 with BOM (Excel-friendly) using the same delimiter
out_lines = [delimiter.join(csv_row) for csv_row in rows]
csv_path.write_text("﻿" + "\n".join(out_lines) + "\n", encoding="utf-8-sig")

print(f"Skipped {len(skipped)} rows: {skipped}")