"""Debug row 2 of review.csv."""
import csv
from pathlib import Path

csv_path = Path(r"C:\code\viethoaphimv4\jobs\Tập 1-9\translation\review.csv")
with csv_path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    print(f"Headers: {reader.fieldnames}")
    for i, row in enumerate(reader, start=2):
        if i <= 4:
            print(f"\nRow {i}:")
            for k, v in row.items():
                print(f"  {k!r}: {v!r}")
        if i == 2:
            from vietdub.translate.csv import import_review_csv
            try:
                import_review_csv(csv_path)
                print("\nFull CSV parsed OK")
            except RuntimeError as e:
                print(f"\nParse error: {e}")
            break