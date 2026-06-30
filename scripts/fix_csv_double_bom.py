"""Remove double UTF-8 BOM from review.csv (caused by utf-8-sig write to file that already had BOM)."""
import csv
from pathlib import Path

csv_path = Path(r"C:\code\viethoaphimv4\jobs\Tập 1-9\translation\review.csv")
raw = csv_path.read_bytes()

# Strip ALL leading BOMs (0xEF 0xBB 0xBF) — there may be one or two
while raw.startswith(b"\xef\xbb\xbf"):
    raw = raw[3:]

# Write back with exactly one BOM
csv_path.write_bytes(b"\xef\xbb\xbf" + raw)

# Validate by re-reading
with csv_path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    print(f"Headers: {fieldnames}")
    print(f"Total rows: {sum(1 for _ in reader)}")