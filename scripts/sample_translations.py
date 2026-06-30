import csv
from pathlib import Path

csv_path = Path(r"C:\code\viethoaphimv4\jobs\Tập 1-9\translation\review.csv")
with csv_path.open(encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")
fallback = [r for r in rows if r["text_vi"].startswith("[CHƯA DỊCH]")]
print(f"Fallback (CHƯA DỊCH) rows: {len(fallback)}")
for fb in fallback:
    print(f"  - {fb['segment_id']}: CN={fb['text_cn'][:60]}")

print("\n--- Sample translations (first 10 rows) ---")
for row in rows[:10]:
    print(f"[{row['segment_id']}] CN: {row['text_cn']}")
    print(f"         VI: {row['text_vi']}")
    print()