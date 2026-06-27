"""
Hiển thị bảng điểm RAGAS V1 vs V2 từ data/ragas_report.json.
Dùng để xem nhanh và chụp ảnh bằng chứng evidence/03_ragas_scores.png.

Cách chạy:
    venv/bin/python src/show_ragas_scores.py
"""
import json
from pathlib import Path

report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
d = json.loads(report_path.read_text(encoding="utf-8"))
v1, v2 = d["prompt_v1_scores"], d["prompt_v2_scores"]

print("=" * 60)
print("  RAGAS Evaluation — V1 (ngắn gọn) vs V2 (có cấu trúc)")
print("=" * 60)
print(f"  {'Metric':22}{'V1':>10}{'V2':>10}   Winner")
print("-" * 60)
for k in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
    a, b = v1.get(k), v2.get(k)
    if a is None or b is None:
        win = "(n/a)"
        sa = "null" if a is None else f"{a:.4f}"
        sb = "null" if b is None else f"{b:.4f}"
    else:
        win = "<- V1" if a > b else ("<- V2" if b > a else "hoa")
        sa, sb = f"{a:.4f}", f"{b:.4f}"
    star = " *" if k == "faithfulness" else ""
    print(f"  {k:22}{sa:>10}{sb:>10}   {win}{star}")
print("-" * 60)
best = max(v1["faithfulness"], v2["faithfulness"])
print(f"  faithfulness cao nhat = {best:.4f}  >= 0.8  -> DAT muc tieu")
print(f"  faithfulness >= 0.9 o CA HAI phien ban -> du dieu kien bonus")
print("=" * 60)
