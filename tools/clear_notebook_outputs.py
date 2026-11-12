"""Xóa đầu ra và số thứ tự chạy trước khi đưa notebook lên Git."""
from pathlib import Path
import json

path = Path(__file__).resolve().parents[1] / "bao-cao-nhom-7.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    cell["metadata"] = {}
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Đã xóa đầu ra: {path.name}")
