"""Kiểu hình thống nhất với slide.pdf, dùng chung cho mã lệnh và notebook.

Mô-đun không chọn backend Matplotlib và không thay dữ liệu tính toán.
Chữ Latin Modern Sans 10 đi cùng toán Computer Modern; có phông dự phòng
để notebook vẫn chạy trên máy chưa cài các phông của TeX.
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.text import Text


RED = "#CC0000"
BLACK = "#000000"
GRAY = "#666666"
COLORS = {"hll": RED, "rusanov": "#CC6600"}
_FONT_FILES = {}


def apply_style():
    """Đặt màu, phông và kiểu xuất; giữ nguyên backend đang dùng."""
    global _FONT_FILES
    _FONT_FILES = {}
    family = "DejaVu Sans"
    try:
        path = Path(font_manager.findfont("Latin Modern Sans", fallback_to_default=False))
        regular = path.parent / "lmsans10-regular.otf"
        if not regular.exists():
            regular = path
        sample = "Độ sâu trung bình ô Tỉ số thời gian Đối chứng Cân bằng áp lực Lưu lượng tuần hoàn Nghiệm số"
        cmap = FT2Font(str(regular)).get_charmap()
        if all(ord(char) in cmap for char in sample):
            family = "Latin Modern Sans"
            _FONT_FILES["regular"] = str(regular)
            for face, suffix in (("bold", "bold"), ("italic", "oblique"),
                                 ("bolditalic", "boldoblique")):
                variant = regular.parent / f"lmsans10-{suffix}.otf"
                if variant.exists():
                    _FONT_FILES[face] = str(variant)
    except ValueError:
        pass

    mpl.rcParams.update({
        "font.family": family, "font.size": 10, "mathtext.fontset": "cm",
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": RED,
        "axes.labelsize": 10, "axes.edgecolor": "#999999",
        "axes.labelcolor": BLACK, "text.color": BLACK,
        "xtick.color": GRAY, "ytick.color": GRAY,
        "grid.color": "#E5E5E5", "grid.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": True, "legend.framealpha": .95,
        "legend.facecolor": "white", "legend.edgecolor": "#D8D8D8",
        # Type 3 nhúng đường viền véc-tơ cùng ánh xạ Unicode. Tránh lỗi
        # khai báo Type 42 cho phông OTF/CFF Latin Modern của Matplotlib.
        "pdf.fonttype": 3, "ps.fonttype": 3,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    return family


def apply_figure_fonts(fig):
    """Ghim cỡ quang học LM Sans 10, giữ cỡ chữ và kiểu đậm/nghiêng."""
    if not _FONT_FILES:
        return
    # Tạo đủ các nhãn vạch chia trước khi gán phông cho từng Text.
    fig.canvas.draw()
    for label in fig.findobj(match=Text):
        properties = label.get_fontproperties().copy()
        weight = properties.get_weight()
        numeric_weight = font_manager.weight_dict.get(weight, 400) if isinstance(weight, str) else weight
        bold = numeric_weight >= 600
        italic = properties.get_style() in ("italic", "oblique")
        face = "bolditalic" if bold and italic else "bold" if bold else "italic" if italic else "regular"
        path = _FONT_FILES.get(face)
        if path:
            properties.set_file(path)
            label.set_fontproperties(properties)
